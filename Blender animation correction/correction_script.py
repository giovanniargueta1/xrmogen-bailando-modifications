import bpy
from mathutils import Vector, Matrix



# Global list to store detected frames
detected_frames = []



def is_armature(obj):
    """Check if the object is an armature by verifying its type."""
    return obj.type == 'ARMATURE'

def get_armature():
    """Attempt to find the armature object in the scene."""
    for obj in bpy.data.objects:
        if is_armature(obj):
            return obj
    return None

def get_bone_bounds(armature, bone_name):
    """Get the bounding box of a specific bone in world space."""
    bone = armature.pose.bones.get(bone_name)
    if not bone:
        print(f"Bone '{bone_name}' not found in armature '{armature.name}'.")
        return None

    # Get bone head and tail in world space
    head_world = armature.matrix_world @ bone.head
    tail_world = armature.matrix_world @ bone.tail

    # Approximate the bounding box
    min_bound = Vector((min(head_world.x, tail_world.x),
                        min(head_world.y, tail_world.y),
                        min(head_world.z, tail_world.z)))
    max_bound = Vector((max(head_world.x, tail_world.x),
                        max(head_world.y, tail_world.y),
                        max(head_world.z, tail_world.z)))
    return min_bound, max_bound

def check_bone_intersection(armature, bone1, bone2):
    """Check if two bones intersect using their bounding boxes."""
    bounds1 = get_bone_bounds(armature, bone1)
    bounds2 = get_bone_bounds(armature, bone2)

    if not bounds1 or not bounds2:
        return False

    min1, max1 = bounds1
    min2, max2 = bounds2
    
   
    # Check for bounding box overlap
    return (min1.x <= max2.x and max1.x >= min2.x and
            min1.y <= max2.y and min1.y >= min2.y and
            min1.z <= max2.z and max1.z >= min2.z)

def highlight_bone(armature, bone_name):
    """Highlight a specific bone by adjusting its viewport display."""
    bone = armature.pose.bones.get(bone_name)
    if not bone:
        return

    # Set custom color for the bone (in pose mode)
    bone.bone.select = True  # Select to visually highlight the bone
    print(f"Bone '{bone_name}' highlighted for collision.")

def reset_bone_highlighting(armature):
    """Reset all bones' highlighting in the armature."""
    for bone in armature.data.bones:
        bone.select = False  # Deselect to remove highlighting
        
def frame_change_handler(scene):
    """Function to check collisions for bones and cylinders."""
    global detected_frames
    
    armature = get_armature()
    if not armature:
        print("No armature object found in the scene.")
        return

    # Delete and recreate cylinders
    for obj in bpy.data.objects:
        if obj.name.startswith("Cylinder_"):
            bpy.data.objects.remove(obj, do_unlink=True)
            
            
     # Individual bones to check for collision
    bone1 = "L_Wrist"  # Replace with the name of the first bone
    bone2 = "R_Wrist"  # Replace with the name of the second bone
    


     # Detect individual bone collisions
    if check_bone_intersection(armature, bone1, bone2):
        print(f"Bone collision detected between {bone1} and {bone2} at frame {scene.frame_current}")
        if scene.frame_current not in detected_frames:
            detected_frames.append(scene.frame_current)
    # Define bone and cylinder pairs
    bone_pairs = [("L_Wrist", "L_Elbow"), ("R_Wrist", "R_Elbow")]
    cylinder_names = [] 
    
    # Create cylinders for bone pairs and check for cylinder collisions
    for bone1, bone2 in bone_pairs:
        bone1_pose = armature.pose.bones.get(bone1)
        bone2_pose = armature.pose.bones.get(bone2)
        if bone1_pose and bone2_pose:
            # Create cylinders between the bones
            start_point = armature.matrix_world @ bone1_pose.head
            end_point = armature.matrix_world @ bone2_pose.head
            cylinder_name = f"Cylinder_{bone1}_{bone2}"
            create_or_update_cylinder(cylinder_name, start_point, end_point)
            cylinder_names.append(cylinder_name)

        
    # Check cylinder collisions based on midpoints
    for i in range(len(cylinder_names)):
        for j in range(i + 1, len(cylinder_names)):
            if check_cylinder_collision_midpoints(cylinder_names[i], cylinder_names[j]):
                print(f"Cylinder collision detected between {cylinder_names[i]} and {cylinder_names[j]} (midpoints) at frame {scene.frame_current}")
                if scene.frame_current not in detected_frames:
                    detected_frames.append(scene.frame_current)


    # Print detected frames
    if scene.frame_current == 1:
        print(f"Collision frames detected so far: {detected_frames}")

def create_or_update_cylinder(name, start_point, end_point, radius=0.0125):
    """
    Create or update a cylinder that connects two points in space.
    :param name: Unique name of the cylinder object.
    :param start_point: Start point of the cylinder in world space.
    :param end_point: End point of the cylinder in world space.
    :param radius: Radius of the cylinder.
    """
    # Create a new cylinder
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=1, location=(0, 0, 0))
    obj = bpy.context.object
    obj.name = name
    obj.data.name = f"{name}_mesh"

    # Calculate the position and orientation of the cylinder
    direction = end_point - start_point
    length = direction.length
    obj.scale = (radius, radius, length / 2)  # Scale the cylinder to fit the length
    obj.location = (start_point + end_point) / 2  # Position at the midpoint
    
    # Align the cylinder to the direction
    direction.normalize()
    rotation_matrix = direction.to_track_quat('Z', 'Y').to_matrix().to_4x4()
    obj.matrix_world = Matrix.Translation(obj.location) @ rotation_matrix
    
def check_cylinder_collision_midpoints(cylinder1, cylinder2, threshold=0.135):
    """Check if two cylinders collide based on their midpoints."""
    obj1 = bpy.data.objects.get(cylinder1)
    obj2 = bpy.data.objects.get(cylinder2)

    if not obj1 or not obj2:
        return False

    # Calculate midpoints for both cylinders
    midpoint1 = obj1.location
    midpoint2 = obj2.location

   

    # Calculate distance between midpoints
    distance = (midpoint1 - midpoint2).length

    # Check if distance is below the threshold
    return distance < threshold

               

def get_cylinder_bounds(cylinder_name):
    """Get the bounding box of a cylinder in world space."""
    obj = bpy.data.objects.get(cylinder_name)
    if not obj:
        return None

    # Calculate the bounding box in world space
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_bound = Vector((min(c.x for c in bbox_corners),
                        min(c.y for c in bbox_corners),
                        min(c.z for c in bbox_corners)))
    max_bound = Vector((max(c.x for c in bbox_corners),
                        max(c.y for c in bbox_corners),
                        max(c.z for c in bbox_corners)))
    return min_bound, max_bound

def check_cylinder_collision(cylinder1, cylinder2):
    """Check if two cylinders collide using their bounding boxes."""
    bounds1 = get_cylinder_bounds(cylinder1)
    bounds2 = get_cylinder_bounds(cylinder2)

    if not bounds1 or not bounds2:
        return False

    min1, max1 = bounds1
    min2, max2 = bounds2

    # Check for bounding box overlap
    return (min1.x <= max2.x and max1.x >= min2.x and
            min1.y <= max2.y and max1.y >= min2.y and
            min1.z <= max2.z and max1.z >= min2.z)

 

def register_update_handler():
    """Attach the frame change handler."""
    if frame_change_handler not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(frame_change_handler)
        print("Frame change handler registered.")

def unregister_update_handler():
    """Remove the frame change handler."""
    if frame_change_handler in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(frame_change_handler)
        print("Frame change handler unregistered.")
        


# Unregister and clean up before re-registering
unregister_update_handler()

# Register the update handler
register_update_handler()

