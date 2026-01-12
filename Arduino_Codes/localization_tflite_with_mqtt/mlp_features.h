#ifndef MLP_FEATURES_H
#define MLP_FEATURES_H

// Total number of RSSI features
#define FEATURE_COUNT 35

// Total number of classes
#define LABEL_COUNT 11

// MAC addresses in the exact order used during training
static const char* mlp_feature_macs[FEATURE_COUNT] = {
    "00:27:E3:29:CD:E0",
    "00:27:E3:29:CE:40",
    "00:27:E3:29:D1:20",
    "00:27:E3:29:D2:A0",
    "00:27:E3:29:D2:E0",
    "00:27:E3:29:DC:00",
    "00:27:E3:29:E8:80",
    "00:A3:8E:4E:29:A1",
    "00:A3:8E:60:EB:E1",
    "00:A3:8E:60:F3:80",
    "00:A3:8E:65:44:00",
    "00:A3:8E:67:4C:81",
    "08:4F:A9:10:4B:81",
    "08:4F:A9:2D:77:41",
    "08:4F:A9:2D:8A:41",
    "08:4F:A9:2D:8A:A1",
    "08:4F:A9:30:00:01",
    "08:4F:A9:39:68:21",
    "08:4F:A9:39:6A:41",
    "40:01:7A:01:B3:82",
    "40:01:7A:01:B8:82",
    "40:01:7A:AF:D3:62",
    "40:01:7A:B1:00:42",
    "4C:77:6D:10:31:E1",
    "70:EA:1A:22:8F:E1",
    "70:EA:1A:22:96:21",
    "70:EA:1A:22:96:A1",
    "70:EA:1A:22:9B:81",
    "70:EA:1A:22:9F:21",
    "70:EA:1A:22:9F:E1",
    "70:EA:1A:22:A5:A1",
    "70:EA:1A:22:A9:41",
    "70:EA:1A:22:A9:A1",
    "70:EA:1A:65:2E:A1",
    "CC:DB:93:65:2D:20"
};

// Label strings in index order (must match LabelEncoder order)
static const char* mlp_label_map[LABEL_COUNT] = {
    "bijuOffice",
    "csHod",
    "d151d153",
    "d171",
    "dlt7",
    "dlt8",
    "iotLab2",
    "itRoom",
    "meetingRoom",
    "roboticsLab",
    "systemsLab2"
};

#endif
