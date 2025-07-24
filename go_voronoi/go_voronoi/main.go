//
// Main package for Voronoi diagram calculations
//

package main

import (
	"fmt"
)


func main() {
	fmt.Println("Starting Voronoi cone generation test...")

	// Create some test points with different spacings
	points := []Point2D{
		NewPoint2D(1.0, 2.0, 0.5, 0.5, 0.5),   // Gray
		NewPoint2D(3.0, 4.0, 0.8, 0.2, 0.1),   // Red
		NewPoint2D(0.5, 0.5, 0.1, 0.8, 0.2),   // Green
		NewPoint2D(5.0, 1.0, 0.2, 0.1, 0.8),   // Blue
	}

	// Print test points
	fmt.Println("Test points:")
	for i, point := range points {
		fmt.Printf("Point%d: %+v\n", i+1, point)
	}

	// Test point density analysis
	fmt.Println("\nPoint density analysis:")
	minSep := MinPointSeparation(points)
	maxSep := MaxPointSeparation(points)
	avgSep := AveragePointSeparation(points)
	
	fmt.Printf("Minimum separation: %.3f\n", minSep)
	fmt.Printf("Maximum separation: %.3f\n", maxSep)
	fmt.Printf("Average separation: %.3f\n", avgSep)

	// Calculate optimal cone parameters
	optimalParams := CalculateOptimalConeParams(points, 350)
	fmt.Printf("\nOptimal cone parameters:\n")
	fmt.Printf("Height: %.3f\n", optimalParams.Height)
	fmt.Printf("Base Radius: %.3f\n", optimalParams.BaseRadius)
	fmt.Printf("Triangles per fan: %d\n", optimalParams.TrianglesPerFan)

	// Test cone construction with optimal parameters
	fmt.Println("\nTesting cone construction with optimal parameters...")
	cone1 := ConstructCone(optimalParams, points[0])
	cone2 := ConstructCone(optimalParams, points[1])
	
	fmt.Printf("Cone1 has %d triangles\n", len(cone1))
	fmt.Printf("Cone2 has %d triangles\n", len(cone2))
	fmt.Printf("Sample triangle from Cone1: %+v\n", cone1[0])

	// Test cone overlaps
	fmt.Println("\nTesting cone overlaps...")
	for i := 0; i < len(points); i++ {
		for j := i + 1; j < len(points); j++ {
			overlaps := ConeOverlaps(points[i], points[j], optimalParams)
			fmt.Printf("Cone %d and Cone %d overlap: %t\n", i+1, j+1, overlaps)
		}
	}

	// Test cone height calculation
	fmt.Println("\nTesting cone height calculation...")
	testPoint := Point2D{X: 1.5, Y: 2.5} // Point between cone 1 and 2
	for i, coneBase := range points {
		height := GetConeHeightAtPoint(testPoint, coneBase, optimalParams)
		fmt.Printf("Height of cone %d at test point (%.1f, %.1f): %.3f\n", i+1, testPoint.X, testPoint.Y, height)
	}

	// Test dominant cone detection
	fmt.Println("\nTesting dominant cone detection...")
	testPoints := []Point2D{
		{X: 1.5, Y: 2.5},  // Between cone 1 and 2
		{X: 2.0, Y: 3.0},  // Center area
		{X: 0.8, Y: 1.8},  // Near cone 1
		{X: 4.5, Y: 1.5},  // Near cone 4
	}
	
	for i, testPt := range testPoints {
		dominantID, maxHeight := FindDominantCone(testPt, points, optimalParams)
		if dominantID >= 0 {
			fmt.Printf("Test point %d (%.1f, %.1f): Cone %d dominates with height %.3f\n", 
				i+1, testPt.X, testPt.Y, dominantID+1, maxHeight)
		} else {
			fmt.Printf("Test point %d (%.1f, %.1f): No cone dominates (outside all cones)\n", 
				i+1, testPt.X, testPt.Y)
		}
	}

	// Test Voronoi grid sampling
	fmt.Println("\nTesting Voronoi grid sampling...")
	
	// Calculate bounds that include all points plus buffer
	minX, maxX := points[0].X, points[0].X
	minY, maxY := points[0].Y, points[0].Y
	for _, p := range points {
		if p.X < minX { minX = p.X }
		if p.X > maxX { maxX = p.X }
		if p.Y < minY { minY = p.Y }
		if p.Y > maxY { maxY = p.Y }
	}
	buffer := 1.0
	bounds := [4]float64{minX - buffer, minY - buffer, maxX + buffer, maxY + buffer}
	
	fmt.Printf("Sampling bounds: [%.1f, %.1f] to [%.1f, %.1f]\n", bounds[0], bounds[1], bounds[2], bounds[3])
	
	// Sample a higher resolution grid for smoother results
	gridSize := 1000  // Increased from 10 for much higher resolution
	grid := SampleVoronoiGrid(points, optimalParams, gridSize, bounds)
	
	fmt.Printf("Generated %dx%d Voronoi grid (%d total samples)\n", gridSize, gridSize, gridSize*gridSize)

	// Generate SVG output
	fmt.Println("\nGenerating SVG output...")
	err := GenerateVoronoiSVG(points, grid, bounds, gridSize, "voronoi_diagram.svg", 1200, 1200)
	if err != nil {
		fmt.Printf("Error generating SVG: %v\n", err)
	} else {
		fmt.Println("SVG saved as 'voronoi_diagram.svg' (1200x1200 pixels)")
	}

	fmt.Println("Test completed!")
}