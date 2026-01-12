//
//  ESP32 Localization + Activity Recognition (Dual TFLite Models)
//
//  Requires:
//    - mlp_model.h / mlp_features.h (localization model)
//    - activity_mlp.h (activity model)
//    - MPU6050 connected via I2C
//

#include <WiFi.h>
#include <Wire.h>
#include <Arduino.h>
#include "MPU6050.h"

// ---------- TFLite Micro ----------
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

using tflite::MicroInterpreter;
using tflite::MicroMutableOpResolver;

// ---------- Localization Headers ----------
#include "mlp_model.h"
#include "mlp_features.h"

// ---------- Activity Model Header ----------
#include "activity_mlp.h"
#include "activity_scaler.h"

//Define for Activity Recognition
#define WINDOW_SIZE 50
#define FEATURES 300

// =====================================================
//                     CONFIGURATION
// =====================================================

const char* TARGET_SSID = "BPGC-NAB";
const float MISSING_RSSI = -105.0f;
const float RF_MIN = -110.0f;
const float RF_MAX = -30.0f;

const int WIFI_PERIOD_MS = 12000;
unsigned long last_wifi_scan = 0;

// MPU sampling
MPU6050 mpu;
unsigned long last_imu_time = 0;
const int IMU_RATE_HZ = 50;       // 50 Hz
const int IMU_PERIOD_MS = 1000/IMU_RATE_HZ;




// =====================================================
//                TENSOR ARENAS FOR TWO MODELS
// =====================================================

// Localization requires more RAM → keep 32 KB
constexpr int kLocArenaSize = 32 * 1024;
static uint8_t loc_arena[kLocArenaSize];

// Activity model is small → 12 KB is enough
constexpr int kActArenaSize = 12 * 1024;
static uint8_t act_arena[kActArenaSize];

// =====================================================
//        Interpreters / Tensors for BOTH MODELS
// =====================================================

// ---------- Localization ----------
const tflite::Model* loc_model = nullptr;
MicroInterpreter* loc_interpreter = nullptr;
TfLiteTensor* loc_input = nullptr;
TfLiteTensor* loc_output = nullptr;

// ---------- Activity ----------
const tflite::Model* act_model = nullptr;
MicroInterpreter* act_interpreter = nullptr;
TfLiteTensor* act_input = nullptr;
TfLiteTensor* act_output = nullptr;

// One resolver shared between both
MicroMutableOpResolver<6> resolver;

// =====================================================
//  Utility Functions
// =====================================================

bool macEquals(const char* a, const char* b){
  for (int i = 0; a[i] && b[i]; ++i){
    char ca=a[i], cb=b[i];
    if (ca>='a'&&ca<='z') ca-=32;
    if (cb>='a'&&cb<='z') cb-=32;
    if (ca!=cb) return false;
  }
  return true;
}

void bssidToStr(uint8_t* bssid, char*out, size_t outlen){
  snprintf(out,outlen,"%02X:%02X:%02X:%02X:%02X:%02X",
           bssid[0],bssid[1],bssid[2],bssid[3],bssid[4],bssid[5]);
}

float scale_rssi(float rssi){
  float x=rssi;
  if (x<RF_MIN) x=RF_MIN;
  if (x>RF_MAX) x=RF_MAX;
  return 2.0f*(x-RF_MIN)/(RF_MAX-RF_MIN)-1.0f;
}

int8_t quantize(float x, float s, int zp){
  int32_t q = lroundf(x/s)+zp;
  if (q<-128) q=-128;
  if (q>127) q=127;
  return (int8_t)q;
}

void setup_ops(){
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddReshape();
}

// =====================================================
//                Setup BOTH TFLite Models
// =====================================================

bool setup_localization(){
  loc_model = tflite::GetModel(mlp_model_tflite);
  static MicroInterpreter static_loc(loc_model, resolver,
                                     loc_arena, kLocArenaSize, nullptr);
  loc_interpreter = &static_loc;
  if (loc_interpreter->AllocateTensors() != kTfLiteOk) return false;
  loc_input = loc_interpreter->input(0);
  loc_output = loc_interpreter->output(0);
  return true;
}

bool setup_activity(){
  act_model = tflite::GetModel(activity_mlp_tflite);
  static MicroInterpreter static_act(act_model, resolver,
                                     act_arena, kActArenaSize, nullptr);
  act_interpreter = &static_act;
  if (act_interpreter->AllocateTensors() != kTfLiteOk) return false;
  act_input = act_interpreter->input(0);
  act_output = act_interpreter->output(0);
  return true;
}

// =====================================================
//                Run Localization and Activity
// =====================================================

void run_localization()
{
    // ====================================================
  //               1) LOCALIZATION MLP
  // ====================================================

  Serial.println("\nSCAN");
  int n = WiFi.scanNetworks(false, true);

  float features[FEATURE_COUNT];
  for(int i=0;i<FEATURE_COUNT;i++) features[i]=MISSING_RSSI;

  for(int i=0;i<n;i++){
    if (WiFi.SSID(i) != TARGET_SSID) continue;
    char mac[18];
    bssidToStr(WiFi.BSSID(i), mac, sizeof(mac));

    for(int k=0;k<FEATURE_COUNT;k++){
      if(macEquals(mac, mlp_feature_macs[k])){
        features[k] = WiFi.RSSI(i);
        break;
      }
    }
  }

  float in_s = loc_input->params.scale;
  int in_zp  = loc_input->params.zero_point;

  int8_t* inbuf = loc_interpreter->typed_input_tensor<int8_t>(0);
  for(int i=0;i<FEATURE_COUNT;i++){
    float sc = scale_rssi(features[i]);
    inbuf[i] = quantize(sc, in_s, in_zp);
  }

  loc_interpreter->Invoke();

  int n_labels = loc_output->dims->data[1];
  int best = 0;
  int8_t* outloc = loc_interpreter->typed_output_tensor<int8_t>(0);
  for(int i=1;i<n_labels;i++){
    if (outloc[i] > outloc[best]) best = i;
  }

  Serial.print("Location = ");
  Serial.println(mlp_label_map[best]);
}

void run_activity()
{
  static float window_buf[WINDOW_SIZE][6];
  static int window_index = 0;
  static bool window_filled = false;

  if (millis() - last_imu_time >= IMU_PERIOD_MS){
    last_imu_time = millis();

    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax,&ay,&az,&gx,&gy,&gz);

    float axg=ax/16384.0, ayg=ay/16384.0, azg=az/16384.0;
    float gxds=gx/131.0,   gyds=gy/131.0,   gzds=gz/131.0;

    // Store into rolling window
    window_buf[window_index][0] = axg;
    window_buf[window_index][1] = ayg;
    window_buf[window_index][2] = azg;
    window_buf[window_index][3] = gxds;
    window_buf[window_index][4] = gyds;
    window_buf[window_index][5] = gzds;

    window_index++;

    if(window_index >= WINDOW_SIZE)
    {
      window_index = 0;
      window_filled = true;
    }

    if(!window_filled) return;

    float input_scale = act_input->params.scale;
    int   zp = act_input->params.zero_point;

    int8_t* actbuf = act_interpreter->typed_input_tensor<int8_t>(0);
    
    int idx = 0;
    int ptr = window_index;
    for(int i = 0; i < WINDOW_SIZE; i++)
    {
      int real_i = (ptr + i) % WINDOW_SIZE;
      for(int j = 0; j < 6; j++)
      {
        float raw = window_buf[real_i][j];

        float norm = (raw - scaler_mean[j])/scaler_std[j];

        float q = (norm / input_scale) + zp;
        if(q < -128) q = -128;
        if (q > 127) q = 127;

        actbuf[idx] = (int8_t)q;
        idx++;
      }
    }

    act_interpreter->Invoke();

    int8_t* actout = act_interpreter->typed_output_tensor<int8_t>(0);
    int bestA = 0;
    for(int i=1;i<3;i++){
      if(actout[i] > actout[bestA]) bestA = i;
    }

    const char* act_labels[] = {"stationary", "walking", "stairs"};

    Serial.print("Activity = ");
    Serial.println(act_labels[bestA]);
  }
}

// =====================================================
//                            SETUP
// =====================================================

void setup(){
  Serial.begin(115200);
  delay(300);

  // WiFi (localization)
  WiFi.mode(WIFI_STA);
  WiFi.disconnect(true);
  delay(200);

  // I2C + MPU
  Wire.begin();
  mpu.initialize();
  if (!mpu.testConnection()){
    Serial.println("MPU connection FAILED!");
  } else {
    Serial.println("MPU ready.");
  }

  setup_ops();
  if (!setup_localization()) Serial.println("Loc model fail!");
  if (!setup_activity()) Serial.println("Act model fail!");
}

// =====================================================
//                     MAIN LOOP
// =====================================================

void loop(){

  if(millis() - last_wifi_scan >= WIFI_PERIOD_MS)
  {
    last_wifi_scan = millis();
    run_localization();
  }

  run_activity();
  

  
}
