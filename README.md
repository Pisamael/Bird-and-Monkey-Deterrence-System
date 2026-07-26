# Autonomous Turret Prototype: Wildlife Deterrence System

An AI-powered, non-lethal deterrent system built within **NVIDIA Isaac Sim**. This project serves as a digital twin for an autonomous pan/tilt turret designed to track specific wildlife (macaque monkeys) and execute a multi-phase deterrence sequence, governed by a strict proximity-based safety interlock to protect human bystanders.

## 🚀 Key Features

* **Custom AI Vision (YOLOv11):** Utilizes a custom-trained neural network (`monkey.pt`) fine-tuned via Transfer Learning specifically for macaque detection, bypassing standard COCO dataset limitations.
* **Physics-Driven Tracking:** The turret mechanism (Base, Yoke, Camera Head) is rigged with `UsdPhysics RevoluteJoints` and driven by real-time mathematical angle calculations for cinematic, stutter-free tracking.
* **Ground-Truth Safety Interlock:** Because the AI is hyper-focused on the target species, the script enforces a mathematical 2.5-meter exclusion zone. If a human (`human1.usdz`) enters this radius, the firing sequence is instantly hard-locked, and visual/audio alarms are triggered.
* **3-Phase Escalation Protocol:** 1. **Acquire:** Solid orange LED indicator.
  2. **Lock-On:** Solid red LED with pulsing warning audio.
  3. **Fire:** High-intensity white strobe flash and loud deterrent sound.

## 📂 Repository Structure

* `standalone_turret_prototype.py`: The core standalone application script containing the simulation environment, turret construction, mathematical tracking loop, and escalation logic.
* `monkey.pt`: The custom YOLOv11 weights file.
* `monkey.usdz` / `human1.usdz`: 3D assets spawned dynamically into the Omniverse environment.
* `.gitignore`: Configured to explicitly ignore the 120,000+ files associated with the local Isaac Sim engine and cache, keeping this repository lightweight.

## ⚙️ Requirements & Installation

1. **NVIDIA Isaac Sim:** This project runs as a standalone application inside Isaac Sim's native Python environment. Ensure Isaac Sim is installed via the Omniverse Launcher.
2. **Clone the Repository:**
   ```bash
   git clone [https://github.com/Pisamael/Bird-and-Monkey-Deterrence-System.git](https://github.com/Pisamael/Bird-and-Monkey-Deterrence-System.git)
   cd Bird-and-Monkey-Deterrence-System
Ensure your file paths inside standalone_turret_prototype.py match your local workspace directory.

💻 How to Run
Do not run this using a standard global Python installation, as it requires Omniverse's customized PyTorch and USD libraries.

Run the script using the Python executable bundled inside your Isaac Sim installation:

Bash
# Example path (adjust to match your specific Isaac Sim installation directory)
& "C:\Users\YOUR_USER\AppData\Local\ov\pkg\isaac-sim-4.2.0\python.bat" standalone_turret_prototype.py
🔮 Future Development Pipeline
Dual-Brain YOLO: Re-integrating the custom model alongside the default COCO model to allow simultaneous detection of both wildlife and humans using pure computer vision.

Sensor Fusion: Implementing Lidar or depth-camera nodes to replace basic 2D trigonometric distance calculations with true 3D spatial awareness.

Hardware Deployment: Exporting the Python logic via ROS2 (Robot Operating System) to communicate with physical Arduino/Raspberry Pi microcontrollers on real-world hardware.

Developed by Shawn David Manjila
