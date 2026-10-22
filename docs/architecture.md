# AI Smart Surveillance System - Architecture & Design Specification

## System Topology & Dataflow

```mermaid
flowchart TD
    subgraph Capture [Video Acquisition Subsystem]
        Src[RTSP / Webcam / File] --> Cap[Threaded VideoCapture]
        Cap --> DropQueue["Queue (Drop-Oldest Strategy)"]
    end

    subgraph Preprocessing [Frame Normalization]
        DropQueue --> Letterbox[Letterbox Aspect-Preserving Resize 640x640]
        Letterbox --> Denoise[Bilateral / Gaussian Denoise]
        Denoise --> CLAHE[Brightness Normalization]
    end

    subgraph Perception [Neural Perception Engine]
        CLAHE --> YOLO["YOLOv8 Inference Engine (CPU / CUDA / MPS)"]
        YOLO --> ScaleCoords[Coordinate Back-Projection to Original Resolution]
        ScaleCoords --> NMS["Vectorized NumPy NMS Post-Filtering"]
    end

    subgraph TrackingEngine [Multi-Object Tracking]
        NMS --> TrackerSelector{"Tracker Selection"}
        TrackerSelector -->|Default| ByteTrack["ByteTrack (Kalman Filter + Hungarian Matching)"]
        TrackerSelector -->|Fallback| Centroid["Euclidean Distance Centroid Tracker"]
        ByteTrack --> TrackState["Persistent Track State (ID, Trajectory 30pts, Dwell Time)"]
        Centroid --> TrackState
    end

    subgraph Analytics [Behavioral Rules Engine]
        TrackState --> R1["Zone Intrusion (Ray Casting)"]
        TrackState --> R2["Loitering (> N seconds in Zone)"]
        TrackState --> R3["Abandoned Object (Stationary + Proximity)"]
        TrackState --> R4["Crowd Surge / Overcrowding (> N persons)"]
        TrackState --> R5["After-Hours Intrusion (Midnight Crossing Window)"]
    end

    subgraph Dispatch [Alert & Evidence Management]
        R1 & R2 & R3 & R4 & R5 --> AlertMgr[Alert Manager]
        AlertMgr --> Snapshots["JPEG Snapshot (.jpg)"]
        AlertMgr --> RingBuffer["Ring Buffer Video Clip (~5s Pre & Post .mp4)"]
        AlertMgr --> SQLite[("SQLite Database (surveillance.db)")]
        AlertMgr --> Notif["Optional Alerts: Beep / Telegram / Email"]
    end

    subgraph Visualization [Rendering & Interfaces]
        TrackState & AlertMgr --> Vis[Surveillance Visualizer]
        Vis --> CLI["OpenCV Live Window / Video File"]
        Vis --> UI["Streamlit Interactive Web Dashboard"]
    end
```

## Key Architectural Principles

1. **Decoupled Asynchronous Video Capture**:
   VideoCapture runs in its own OS thread. A size-constrained queue (default size 2) immediately discards stale frames when inference is running. This eliminates buffer lag and guarantees that incoming frames reflect real-world time.

2. **Bidirectional Coordinate Inversion**:
   Frames are normalized to square letterbox dimensions for convolutional inference, and resulting coordinates are mathematically inverted to match native camera coordinates before bounding boxes, polygon checks, or trails are evaluated.

3. **Multi-Object State Lifecycle**:
   Every tracked object maintains an anchor position, last movement time, total dwell time, and a circular deque containing the last 30 foot positions. This enables trajectory trail rendering and stationary duration calculation without global coordinate bloat.

4. **Circular Evidence Ring Buffer**:
   Video clips are not simply started at the moment of an alert. A 300-frame circular ring buffer continuously caches recent frames, so when an incident triggers, the preceding 5 seconds are seamlessly prepended to the following 5 seconds to provide full investigative context.
