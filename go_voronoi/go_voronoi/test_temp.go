package main

import "fmt"

func TestBoundaryFunctions() {
	points := []Point2D{
		NewPoint2D(1.0, 2.0, 0.5, 0.5, 0.5),
	}
	
	coneParams := ConePars{
		Height:          2.0,
		BaseRadius:      1.0,
		TrianglesPerFan: 8,
	}
	
	// Test if functions are accessible
	testPoint := Point2D{X: 1.0, Y: 2.0}
	height := GetConeHeightAtPoint(testPoint, points[0], coneParams)
	fmt.Printf("Height: %f\n", height)
	
	dominantID, maxHeight := FindDominantCone(testPoint, points, coneParams)
	fmt.Printf("Dominant: %d, Height: %f\n", dominantID, maxHeight)
	
	bounds := [4]float64{0, 0, 5, 5}
	grid := SampleVoronoiGrid(points, coneParams, 5, bounds)
	fmt.Printf("Grid size: %d\n", len(grid))
}
