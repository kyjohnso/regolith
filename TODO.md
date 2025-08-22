# Regolith - Lunar Simulation Engine TODO

## Project Overview
Building a game/simulation engine using Rust and Bevy 0.16.1 that allows players to move around a realistic lunar surface and interact with regolith spheroids. The goal is to explore the remnants of a moon-wide compute farm and use regolith to rebuild, adapt, and evolve it.

**Technical Goals:**
- Explore limits of Bevy ECS
- Implement GPU-based physics for regolith simulation
- Target 1M+ particles with real-time physics at 30fps minimum
- Create sandbox for lunar computation exploration

## Development Progress

### Phase 1: Foundation & Basic Setup
- [x] Set up Rust project with Bevy 0.16.1 and basic dependencies
- [x] Create basic Bevy app with 3D scene and orbit camera controls
- [x] Test: Run cargo run to verify basic app launches and camera works
- [x] Add lunar surface terrain (simple plane or heightmap)
- [x] Test: Run cargo run to verify terrain renders correctly

### Phase 2: Player Movement & Interaction
- [x] Implement basic player movement with lunar gravity (1/6th Earth)
- [x] Test: Run cargo run to verify player movement feels right

### Phase 3: Basic Particle System
- [x] Add simple regolith particle spawning system (start with ~1000 particles)
- [x] Create basic particle rendering system with instanced meshes
- [x] Test: Run cargo run to verify particles spawn and render
- [x] Implement CPU-based sphere-sphere collision detection for validation
- [x] Test: Run cargo run to verify collision detection works

### Phase 4: GPU Compute Architecture ✅ **COMPLETED**
- [x] Design GPU compute shader architecture for particle physics
- [x] Implement wgpu compute shader for particle position updates (Infrastructure only)
- [x] Complete GPU-CPU data synchronization for full GPU physics pipeline
- [x] Test: Run cargo run to verify GPU compute integration (GPU physics working)
- [x] Add GPU-based spatial hashing for efficient collision detection
- [x] Implement GPU sphere-sphere collision response system
- [x] Test: Run cargo run to verify GPU physics pipeline

**🎉 MAJOR MILESTONE ACHIEVED**: Full GPU compute pipeline with particle-particle collisions working at 6000 particles with good performance!

### Phase 5: Performance Scaling
- [ ] Scale up to 10k particles and performance test
- [ ] Test: Run cargo run and profile performance at 10k particles
- [ ] Optimize GPU memory management and buffer strategies
- [ ] Scale up to 100k particles with performance profiling
- [ ] Test: Run cargo run and measure fps at 100k particles

### Phase 6: Advanced Physics & Interactions
- [ ] Implement particle interaction with terrain/surface
- [ ] Test: Run cargo run to verify terrain-particle interactions
- [ ] Add basic particle clustering and cohesion effects
- [ ] Scale up to 1M+ particles with target 30fps performance
- [ ] Test: Run cargo run and validate 1M+ particle performance

### Phase 7: Game Mechanics
- [x] Add particle size variation and basic shape differentiation
- [x] Implement player-regolith interaction mechanics
- [x] Test: Run cargo run to verify player can interact with particles
- [x] **Advanced player-particle collision system with realistic physics**
- [ ] Create foundation for compute farm remnants system
- [ ] Add basic regolith manipulation tools for rebuilding

## Technical Architecture

### Core Systems ✅ **FULLY IMPLEMENTED**
- **ECS Architecture**: Leveraging Bevy's Entity Component System
- **Rendering**: Instanced mesh rendering for particles
- **Physics**: GPU compute shaders with wgpu ✅ **WORKING**
- **Collision Detection**: Spatial hashing on GPU ✅ **WORKING**
- **Camera**: Pan-orbit camera for exploration
- **Particle-Particle Interactions**: Full collision response system ✅ **NEW**
- **Player-Particle Physics**: Dynamic collision detection and response ✅ **NEW**

### Performance Targets
- **Particle Count**: 1M+ particles
- **Frame Rate**: 30fps minimum
- **Platform**: Desktop (Windows/Linux/macOS)
- **GPU**: wgpu compute shaders for physics

### Key Dependencies
- `bevy = "0.16.1"`
- `bevy_panorbit_camera = "0.28.0"`
- `rand = "0.8"`

## Recent Achievements 🚀

### December 2024 - GPU Physics Pipeline Complete
- ✅ **Full GPU Compute Pipeline**: Complete infrastructure with spatial hashing
- ✅ **Particle-Particle Collisions**: Realistic sphere-sphere collision detection and response
- ✅ **Player-Particle Interactions**: Dynamic collision system with momentum transfer
- ✅ **Performance**: 6000 particles running smoothly at ~30 FPS
- ✅ **Spatial Hashing**: 64x64x64 uniform grid for efficient collision detection
- ✅ **Physics Simulation**: Gravity, ground collision, damping, friction all working

### Technical Implementation Details
- **GPU Shaders**: WGSL compute shaders for particle physics
- **Buffer Management**: Proper GPU-CPU data synchronization
- **Collision Response**: Realistic impulse-based physics with restitution and friction
- **Memory Management**: Efficient buffer usage with 64-byte alignment
- **System Architecture**: Bevy render app pipeline with chained systems

## Notes
- ✅ Sphere-sphere collisions and basic gravity **COMPLETED**
- ✅ GPU-based physics for scalability **IMPLEMENTED**
- 🎯 **Next Goal**: Scale to 10k+ particles and optimize performance
- Regular testing with `cargo run` after each major addition