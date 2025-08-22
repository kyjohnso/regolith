// Spatial hash compute shader for efficient particle collision detection
// This shader implements a 3D spatial hash grid to reduce collision checks from O(n²) to O(n)

struct Particle {
    position: vec3<f32>,
    _padding1: f32,
    velocity: vec3<f32>,
    _padding2: f32,
    radius: f32,
    mass: f32,
    _padding3: vec2<f32>,
}

struct HashCell {
    particle_count: u32,
    particle_indices: array<u32, 32>, // Max 32 particles per cell
    _padding: array<u32, 3>,
}

struct SpatialHashUniforms {
    particle_count: u32,
    grid_size: vec3<u32>,
    cell_size: f32,
    world_min: vec3<f32>,
    world_max: vec3<f32>,
    max_particles_per_cell: u32,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> hash_grid: array<HashCell>;
@group(0) @binding(2) var<uniform> uniforms: SpatialHashUniforms;

// Hash function for 3D coordinates
fn hash_position(pos: vec3<f32>) -> u32 {
    // Convert world position to grid coordinates
    let grid_pos = vec3<u32>(
        u32(clamp((pos.x - uniforms.world_min.x) / uniforms.cell_size, 0.0, f32(uniforms.grid_size.x - 1u))),
        u32(clamp((pos.y - uniforms.world_min.y) / uniforms.cell_size, 0.0, f32(uniforms.grid_size.y - 1u))),
        u32(clamp((pos.z - uniforms.world_min.z) / uniforms.cell_size, 0.0, f32(uniforms.grid_size.z - 1u)))
    );
    
    // Convert 3D grid coordinates to 1D hash
    return grid_pos.x + grid_pos.y * uniforms.grid_size.x + grid_pos.z * uniforms.grid_size.x * uniforms.grid_size.y;
}

// Clear hash grid (first pass)
@compute @workgroup_size(64, 1, 1)
fn clear_grid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let total_cells = uniforms.grid_size.x * uniforms.grid_size.y * uniforms.grid_size.z;
    
    if (index >= total_cells) {
        return;
    }
    
    // Reset particle count for this cell
    hash_grid[index].particle_count = 0u;
}

// Populate hash grid with particles (second pass)
@compute @workgroup_size(64, 1, 1)
fn populate_grid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_index = global_id.x;
    
    if (particle_index >= uniforms.particle_count) {
        return;
    }
    
    let particle = particles[particle_index];
    let cell_hash = hash_position(particle.position);
    
    // Add particle to cell (simplified without atomics for now)
    let current_count = hash_grid[cell_hash].particle_count;
    
    // Only add if there's space in the cell
    if (current_count < uniforms.max_particles_per_cell) {
        hash_grid[cell_hash].particle_indices[current_count] = particle_index;
        hash_grid[cell_hash].particle_count = current_count + 1u;
    }
}

// Particle collision detection using spatial hash (third pass)
@compute @workgroup_size(64, 1, 1)
fn detect_collisions(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_index = global_id.x;
    
    if (particle_index >= uniforms.particle_count) {
        return;
    }
    
    var particle = particles[particle_index];
    let particle_pos = particle.position;
    
    // Get grid coordinates for this particle
    let grid_pos = vec3<i32>(
        i32((particle_pos.x - uniforms.world_min.x) / uniforms.cell_size),
        i32((particle_pos.y - uniforms.world_min.y) / uniforms.cell_size),
        i32((particle_pos.z - uniforms.world_min.z) / uniforms.cell_size)
    );
    
    var collision_force = vec3<f32>(0.0);
    var collision_count = 0u;
    
    // Check neighboring cells (3x3x3 = 27 cells)
    for (var dx = -1; dx <= 1; dx++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dz = -1; dz <= 1; dz++) {
                let neighbor_pos = grid_pos + vec3<i32>(dx, dy, dz);
                
                // Bounds check
                if (neighbor_pos.x < 0 || neighbor_pos.y < 0 || neighbor_pos.z < 0 ||
                    neighbor_pos.x >= i32(uniforms.grid_size.x) ||
                    neighbor_pos.y >= i32(uniforms.grid_size.y) ||
                    neighbor_pos.z >= i32(uniforms.grid_size.z)) {
                    continue;
                }
                
                // Convert to cell hash
                let cell_hash = u32(neighbor_pos.x) + u32(neighbor_pos.y) * uniforms.grid_size.x + 
                               u32(neighbor_pos.z) * uniforms.grid_size.x * uniforms.grid_size.y;
                
                let cell = hash_grid[cell_hash];
                let particle_count = cell.particle_count;
                
                // Check collisions with particles in this cell
                for (var i = 0u; i < min(particle_count, uniforms.max_particles_per_cell); i++) {
                    let other_index = cell.particle_indices[i];
                    
                    // Skip self-collision
                    if (other_index == particle_index) {
                        continue;
                    }
                    
                    let other_particle = particles[other_index];
                    let distance_vec = particle_pos - other_particle.position;
                    let distance = length(distance_vec);
                    let min_distance = particle.radius + other_particle.radius;
                    
                    // Check for collision
                    if (distance < min_distance && distance > 0.0) {
                        let normal = distance_vec / distance;
                        let overlap = min_distance - distance;
                        
                        // Calculate separation force (stronger for more overlap)
                        let separation_force = normal * overlap * 0.5;
                        collision_force += separation_force;
                        collision_count++;
                        
                        // Apply collision response
                        let relative_velocity = particle.velocity - other_particle.velocity;
                        let velocity_along_normal = dot(relative_velocity, normal);
                        
                        if (velocity_along_normal > 0.0) {
                            continue; // Objects separating
                        }
                        
                        // Calculate collision impulse
                        let restitution = 0.3;
                        let impulse_scalar = -(1.0 + restitution) * velocity_along_normal;
                        let mass_ratio = particle.mass / (particle.mass + other_particle.mass);
                        let impulse = normal * impulse_scalar * (1.0 - mass_ratio);
                        
                        particle.velocity += impulse;
                    }
                }
            }
        }
    }
    
    // Apply accumulated collision forces
    if (collision_count > 0u) {
        particle.position += collision_force / f32(collision_count);
        particle.velocity *= 0.95; // Collision damping
    }
    
    // Write back updated particle
    particles[particle_index] = particle;
}