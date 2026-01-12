# generate_cc_array.py
data = open("mlp_model.tflite", "rb").read()

with open("mlp_model.cc", "w") as f:
    f.write("#include <stdint.h>\n\n")
    f.write("#include <mlp_model.h>\n\n")
    f.write("const unsigned char mlp_model_tflite[] = {\n")

    for i, b in enumerate(data):
        if i % 12 == 0:
            f.write("  ")
        f.write(f"0x{b:02X}, ")
        if i % 12 == 11:
            f.write("\n")
    
    f.write("};\n\n")
    f.write(f"const unsigned int mlp_model_tflite_len = {len(data)};\n")

print("Generated mlp_model.cc successfully.")