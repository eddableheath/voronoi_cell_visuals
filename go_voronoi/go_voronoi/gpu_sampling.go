//
// GPU-accelerated Voronoi grid sampling using OpenGL compute shaders
//

package main

import (
	"fmt"
	"runtime"

	"github.com/go-gl/gl/v4.1-core/gl"
	"github.com/go-gl/glfw/v3.3/glfw"
)

// GPUSampler handles GPU-accelerated Voronoi sampling
type GPUSampler struct {
	// The compiled compute shader program
	program uint32
	
	// GPU memory buffers (SSBOs - Shader Storage Buffer Objects)
	pointsSSBO  uint32  // For storing point data (x, y, color)
	paramsSSBO  uint32  // For cone parameters (height, radius, etc.)
	boundsSSBO  uint32  // For bounds and grid size
	resultSSBO  uint32  // For results from GPU
	
	// Track if we're initialized
	initialized bool
}

// Initialize sets up the GPU sampler
func (gpu *GPUSampler) Initialize() error {
	// Lock OS thread for OpenGL context
	runtime.LockOSThread()

	// Step 1: Initialize GLFW
	if err := glfw.Init(); err != nil {
		return fmt.Errorf("failed to initialize GLFW: %w", err)
	}

	// Step 2: Set OpenGL version requirements (4.1 for macOS compatibility)
	glfw.WindowHint(glfw.ContextVersionMajor, 4)
	glfw.WindowHint(glfw.ContextVersionMinor, 1)
	glfw.WindowHint(glfw.OpenGLProfile, glfw.OpenGLCoreProfile)
	glfw.WindowHint(glfw.OpenGLForwardCompatible, glfw.True) // Required for macOS
	glfw.WindowHint(glfw.Visible, glfw.False) // Hidden window for compute only

	// Step 3: Create minimal window for OpenGL context
	window, err := glfw.CreateWindow(1, 1, "GPU Compute", nil, nil)
	if err != nil {
		glfw.Terminate()
		return fmt.Errorf("failed to create window: %w", err)
	}
	window.MakeContextCurrent()

	// Step 4: Initialize OpenGL
	if err := gl.Init(); err != nil {
		return fmt.Errorf("failed to initialize OpenGL: %w", err)
	}

	// Check OpenGL version and extensions
	version := gl.GoStr(gl.GetString(gl.VERSION))
	renderer := gl.GoStr(gl.GetString(gl.RENDERER))
	fmt.Printf("OpenGL Version: %s\n", version)
	fmt.Printf("GPU Renderer: %s\n", renderer)
	
	// Check for compute shader extension (needed for OpenGL < 4.3)
	extensions := gl.GoStr(gl.GetString(gl.EXTENSIONS))
	_ = extensions // We'll use this later if needed
	fmt.Printf("Checking compute shader support...\n")

	// Step 5: Check GPU capabilities and compute shader support
	var maxWorkGroupSize [3]int32
	gl.GetIntegeri_v(gl.MAX_COMPUTE_WORK_GROUP_SIZE, 0, &maxWorkGroupSize[0])
	gl.GetIntegeri_v(gl.MAX_COMPUTE_WORK_GROUP_SIZE, 1, &maxWorkGroupSize[1])
	gl.GetIntegeri_v(gl.MAX_COMPUTE_WORK_GROUP_SIZE, 2, &maxWorkGroupSize[2])
	
	fmt.Printf("GPU Compute capabilities: %dx%dx%d max work group size\n", 
		maxWorkGroupSize[0], maxWorkGroupSize[1], maxWorkGroupSize[2])

	// Check if compute shaders are actually supported
	if maxWorkGroupSize[0] == 0 || maxWorkGroupSize[1] == 0 || maxWorkGroupSize[2] == 0 {
		return fmt.Errorf("compute shaders not supported on this GPU/OpenGL version")
	}

	// Additional compute shader support checks
	var maxComputeWorkGroupInvocations int32
	gl.GetIntegerv(gl.MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &maxComputeWorkGroupInvocations)
	
	var maxComputeShaderStorageBlocks int32  
	gl.GetIntegerv(gl.MAX_COMPUTE_SHADER_STORAGE_BLOCKS, &maxComputeShaderStorageBlocks)
	
	fmt.Printf("Max compute invocations: %d\n", maxComputeWorkGroupInvocations)
	fmt.Printf("Max storage blocks: %d\n", maxComputeShaderStorageBlocks)
	
	if maxComputeShaderStorageBlocks < 4 {
		return fmt.Errorf("insufficient storage blocks for our algorithm (need 4, got %d)", maxComputeShaderStorageBlocks)
	}

	// Step 6: Create GPU buffers
	gl.GenBuffers(1, &gpu.pointsSSBO)
	gl.GenBuffers(1, &gpu.paramsSSBO)
	gl.GenBuffers(1, &gpu.boundsSSBO)
	gl.GenBuffers(1, &gpu.resultSSBO)

	// Note: We'll create the compute shader later - it's complex!
	
	gpu.initialized = true
	return nil
}

// TestGPUSetup tests if we can initialize the GPU context
func TestGPUSetup() {
	fmt.Println("Testing GPU setup...")
	
	// Create GPU sampler
	gpu := &GPUSampler{}
	
	// Try to initialize
	err := gpu.Initialize()
	if err != nil {
		fmt.Printf("❌ GPU initialization failed: %v\n", err)
		return
	}
	
	fmt.Println("✅ GPU initialization successful!")
	fmt.Printf("✅ OpenGL context created\n")
	fmt.Printf("✅ %d GPU buffers created\n", 4) // points, params, bounds, result
	
	// Clean up
	gpu.Cleanup()
	fmt.Println("✅ GPU cleanup completed")
}

// Cleanup releases GPU resources
func (gpu *GPUSampler) Cleanup() {
	if gpu.initialized {
		// Delete GPU buffers
		gl.DeleteBuffers(1, &gpu.pointsSSBO)
		gl.DeleteBuffers(1, &gpu.paramsSSBO)
		gl.DeleteBuffers(1, &gpu.boundsSSBO)
		gl.DeleteBuffers(1, &gpu.resultSSBO)
		
		// Delete shader program if it exists
		if gpu.program != 0 {
			gl.DeleteProgram(gpu.program)
		}
		
		// Terminate GLFW
		glfw.Terminate()
		
		gpu.initialized = false
		fmt.Println("GPU resources cleaned up")
	}
}
