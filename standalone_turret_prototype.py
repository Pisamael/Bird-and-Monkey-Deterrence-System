import numpy as np
import winsound
import math

# --- Start Isaac Sim Standalone App ---
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
import isaacsim.core.utils.prims as prims_utils
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.utils.rotations import euler_angles_to_quat
from pxr import UsdGeom, UsdPhysics, Gf, UsdLux

# Initialize World
world = World(stage_units_in_meters=1.0)
stage = stage_utils.get_current_stage()

print("[INFO] Loading Demo Environment...")

# --- A. LOAD APARTMENT (WITH FALLBACK) ---
try:
    from isaacsim.storage.native import get_assets_root_path
    assets_root_path = get_assets_root_path()
    room_usd_path = assets_root_path + "/Isaac/Environments/Simple_Room/simple_room.usd"
    prims_utils.create_prim("/World/Apartment", usd_path=room_usd_path)
except:
    print("🚧 [FALLBACK] Cloud assets unavailable. Spawning a flat ground plane instead!")
    world.scene.add_default_ground_plane()

sun = UsdLux.DomeLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(800.0)

# --- B to G. BUILD TURRET HARDWARE ---
base_path = "/World/TurretBase"
prims_utils.create_prim(base_path, "Cylinder", position=np.array([0.0, 0.0, 0.1]), scale=np.array([0.3, 0.3, 0.2]))
UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(base_path))
UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(base_path))
UsdPhysics.FixedJoint.Define(stage, "/World/BaseFixedJoint").CreateBody1Rel().SetTargets([base_path])

yoke_path = "/World/TurretYoke"
prims_utils.create_prim(yoke_path, "Cube", position=np.array([0.0, 0.0, 0.35]), scale=np.array([0.15, 0.15, 0.15]))
UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(yoke_path))
UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(yoke_path)).CreateMassAttr().Set(1.0) 

pan_joint_path = "/World/PanJoint"
pan_joint = UsdPhysics.RevoluteJoint.Define(stage, pan_joint_path)
pan_joint.CreateBody0Rel().SetTargets([base_path])
pan_joint.CreateBody1Rel().SetTargets([yoke_path])
pan_joint.CreateAxisAttr("Z")
pan_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 1.25)) 
pan_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))  
pan_drive = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(pan_joint_path), "angular")
pan_drive.CreateStiffnessAttr(10000.0) # Increased stiffness for snappy tracking
pan_drive.CreateDampingAttr(1000.0)

head_path = "/World/CameraHead"
prims_utils.create_prim(head_path, "Cube", position=np.array([0.0, 0.0, 0.55]), scale=np.array([0.25, 0.12, 0.12]))
UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(head_path))
UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(head_path)).CreateMassAttr().Set(1.0) 

barrel_path = "/World/CameraHead/GunBarrel"
prims_utils.create_prim(barrel_path, "Cylinder", position=np.array([0.2, 0.0, 0.0]), scale=np.array([0.05, 0.05, 0.2]))
barrel_prim = stage.GetPrimAtPath(barrel_path)
UsdGeom.Xformable(barrel_prim).AddRotateXYZOp().Set(Gf.Vec3d(0, 90, 0))

# ADD THE CAMERA SO YOU CAN VIEW IT IN THE UI
camera_path = "/World/CameraHead/TrackingCamera"
camera_prim = UsdGeom.Camera.Define(stage, camera_path)
camera_prim.CreateFocalLengthAttr().Set(24.0)
camera_prim.CreateFocusDistanceAttr().Set(400.0)
UsdGeom.Xformable(camera_prim).AddTranslateOp().Set(Gf.Vec3d(0.2, 0.0, 0.05)) # Positioned slightly above the barrel
UsdGeom.Xformable(camera_prim).AddRotateXYZOp().Set(Gf.Vec3d(90, 0, 90)) # Aligned to look forward

strobe_path = "/World/CameraHead/StrobeLight"
strobe_light = UsdLux.SphereLight.Define(stage, strobe_path)
strobe_light.CreateRadiusAttr(0.1)
strobe_light.CreateIntensityAttr(0.0) 

tilt_joint_path = "/World/TiltJoint"
tilt_joint = UsdPhysics.RevoluteJoint.Define(stage, tilt_joint_path)
tilt_joint.CreateBody0Rel().SetTargets([yoke_path])
tilt_joint.CreateBody1Rel().SetTargets([head_path])
tilt_joint.CreateAxisAttr("Y")
tilt_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 1.3333)) 
tilt_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))  
tilt_drive = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(tilt_joint_path), "angular")
tilt_drive.CreateStiffnessAttr(10000.0)
tilt_drive.CreateDampingAttr(1000.0)

# --- H. ASSET LOADING (Updated to human1.usdz) ---
monkey_orientation = euler_angles_to_quat(np.array([90.0, -90.0, 0.0]), degrees=True)
prims_utils.create_prim(
    prim_path="/World/Monkey1", 
    usd_path="Z:/Dev_Workspace/isaacsim/monkey.usdz",
    position=np.array([2.5, 0.0, 0.4]), 
    orientation=monkey_orientation,
    scale=np.array([0.01, 0.01, 0.01]) 
)

# Hardcoded Human Position for distance checking
human_x = 2.92
human_y = 3.44
human_orientation = euler_angles_to_quat(np.array([0.0, 0.0, -50.0]), degrees=True)
prims_utils.create_prim(
    prim_path="/World/Human", 
    usd_path="Z:/Dev_Workspace/isaacsim/human1.usdz", # <-- Updated to human1!
    position=np.array([human_x, human_y, -0.78978]), 
    orientation=human_orientation, 
    scale=np.array([1.0, 1.0, 1.0])
)

world.reset()

monkey_angle = 0.0 
pause_timer = 0 
fire_sequence = 0
step_count = 0

print("[SUCCESS] DEMO MODE ACTIVE. Cameras attached, console logging finalized.")
print("===============================================================")

# --- I. FAKE TRACKING LOOP ---
while simulation_app.is_running():
    world.step(render=True)
    step_count += 1
    
    if world.is_playing():
        
        # 1. ORBIT & WAYPOINT LOGIC
        if pause_timer > 0:
            pause_timer -= 1 
        else:
            monkey_speed = 0.008 
            monkey_angle += monkey_speed 
            current_angle_mod = monkey_angle % 6.28 
            
            if abs(current_angle_mod - 0.86) < 0.01:
                pause_timer = 180 
                monkey_angle += 0.02 
            elif abs(current_angle_mod - 3.5) < 0.01:
                pause_timer = 140 
                monkey_angle += 0.02 
                
        # Move the Monkey
        new_x = 2.5 * math.cos(monkey_angle)
        new_y = 2.5 * math.sin(monkey_angle)
        stage.GetPrimAtPath("/World/Monkey1").GetAttribute("xformOp:translate").Set(Gf.Vec3d(new_x, new_y, 0.4))

        # 2. GROUND TRUTH PERFECT TRACKING
        pan_target = math.degrees(monkey_angle) 
        tilt_target = 8.0 
        
        pan_drive.GetTargetPositionAttr().Set(pan_target)
        tilt_drive.GetTargetPositionAttr().Set(tilt_target)

        # 3. SAFETY INTERLOCK & ESCALATION LOGIC
        distance_to_human = math.sqrt((new_x - human_x)**2 + (new_y - human_y)**2)
        
        if distance_to_human < 2.5:
            # SAFETY ON: Monkey and Human are together
            
            # Print safety warning clearly to CMD (every 60 frames so it is readable)
            if step_count % 60 == 0:
                print("🛑 [SAFETY INTERLOCK] Human in view! Weapons disabled.")
            
            if step_count % 20 < 10:
                strobe_light.GetColorAttr().Set(Gf.Vec3f(1.0, 0.0, 0.0)) 
            else:
                strobe_light.GetColorAttr().Set(Gf.Vec3f(0.0, 0.0, 1.0)) 
            strobe_light.GetIntensityAttr().Set(50000.0)
            
            if step_count % 40 == 0:
                winsound.Beep(800, 150) 
                
            fire_sequence = 0 # Crucial: Locks out the shooting sequence completely
            
        else:
            # SAFETY OFF: 3-Phase Firing Sequence
            
            # Print exact phases to CMD exactly when they start
            if fire_sequence == 0:
                print("\n⚠️ [PHASE 1] Target Acquired: Light only.")
            elif fire_sequence == 60:
                print("\n🚨 [PHASE 2] Locking On: Light & Sound.")
            
            fire_sequence += 1
            
            if fire_sequence < 60:
                # Phase 1: Light (Target Acquired)
                strobe_light.GetColorAttr().Set(Gf.Vec3f(1.0, 0.4, 0.0)) 
                strobe_light.GetIntensityAttr().Set(20000.0)
                
            elif fire_sequence < 120:
                # Phase 2: Light + Sound (Locking On)
                strobe_light.GetColorAttr().Set(Gf.Vec3f(1.0, 0.0, 0.0)) 
                strobe_light.GetIntensityAttr().Set(40000.0)
                if step_count % 15 == 0:
                    winsound.Beep(1200, 50) 
                    
            else:
                # Phase 3: Shoot
                print("💥 [PHASE 3] SHOT FIRED!")
                strobe_light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0)) 
                strobe_light.GetIntensityAttr().Set(200000.0) 
                winsound.Beep(2500, 200) 
                fire_sequence = 0 # Loop back to Phase 1

simulation_app.close()