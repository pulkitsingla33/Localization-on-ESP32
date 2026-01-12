#include <WiFi.h>
#include <PubSubClient.h>
#include "WiFiClientSecure.h"


const char* ssid = "yourssid";
const char* password = "yourpasswd";

const char* mqtt_server = "a2b6b9feb01b4f678b5891f1f5720153.s1.eu.hivemq.cloud"; // e.g. random-1234.s1.eu.hivemq.cloud
const int   mqtt_port = 8883;

const char* mqtt_username = "pulkit_IoT";
const char* mqtt_password = "pulkit_IoT123";

WiFiClientSecure espClient;
PubSubClient client(espClient);

void setup_wifi() {
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("WiFi connected");
}

void reconnect() {
  if (client.connected()) return;

  static unsigned long lastAttempt = 0;
  if (millis() - lastAttempt < 5000) return;  // retry every 5s
  lastAttempt = millis();

  Serial.print("Reconnecting MQTT... ");
  
  if (client.connect("esp32-client", mqtt_username, mqtt_password)) {
    Serial.println("connected!");
    client.subscribe("android/test");
  } 
  else {
    Serial.print("failed, rc=");
    Serial.println(client.state());
  }
}


void setup() {
  Serial.begin(115200);
  setup_wifi();

  // IMPORTANT
  espClient.setInsecure();   // No certificate needed
  
  client.setServer(mqtt_server, mqtt_port);
}

void loop() {
  if (!client.connected()) reconnect();
  client.loop();

  static unsigned long last = 0;
  if (millis() - last > 5000) {
    last = millis();
    client.publish("esp32/test", "hello from esp32");
  }
}
