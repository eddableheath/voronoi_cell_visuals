// Geometry functions for Voronoi diagrams
package main

// Distance calculates the Euclidean distance between two 2D points
import (
	"math"
)

// Distance calculates the Euclidean distance between two 2D points
func Distance(p1, p2 Point2D) float64 {
	return math.Sqrt((p2.X-p1.X)*(p2.X-p1.X) + (p2.Y-p1.Y)*(p2.Y-p1.Y))
}

// MaxPointSeparation calculates the maximum distance between any two points
func MaxPointSeparation(points []Point2D) float64 {
	if len(points) < 2 {
		return 0.0
	}

	maxDist := 0.0
	for i := range points {
		for j := i + 1; j < len(points); j++ {
			dist := Distance(points[i], points[j])
			if dist > maxDist {
				maxDist = dist
			}
		}
	}
	return maxDist
}

// MinPointSeparation calculates the minimum distance between any two points
func MinPointSeparation(points []Point2D) float64 {
	if len(points) < 2 {
		return 0.0
	}

	minDist := math.Inf(1) // Start with positive infinity
	for i := 0; i < len(points); i++ {
		for j := i + 1; j < len(points); j++ {
			dist := Distance(points[i], points[j])
			if dist < minDist {
				minDist = dist
			}
		}
	}
	return minDist
}

// AveragePointSeparation calculates the average distance between all point pairs
func AveragePointSeparation(points []Point2D) float64 {
	if len(points) < 2 {
		return 0.0
	}

	totalDist := 0.0
	count := 0
	for i := 0; i < len(points); i++ {
		for j := i + 1; j < len(points); j++ {
			totalDist += Distance(points[i], points[j])
			count++
		}
	}
	return totalDist / float64(count)
}

// CalculateOptimalConeParams determines cone height and radius based on point density
func CalculateOptimalConeParams(points []Point2D, trianglesPerFan int) ConePars {
	if len(points) < 2 {
		// Default parameters for single point or empty set
		return ConePars{
			Height:          2.0,
			BaseRadius:      5.0, // Larger default radius
			TrianglesPerFan: trianglesPerFan,
		}
	}

	// Find the maximum distance from any point to any other point
	maxDist := MaxPointSeparation(points)
	
	// Base radius should be large enough to ensure complete coverage
	// Use a generous multiplier to ensure even boundary areas are covered
	baseRadius := maxDist * 0.8

	// Height should be proportional to the radius to maintain good cone shape
	height := baseRadius * 1.5

	// Ensure minimum viable dimensions
	if baseRadius < 0.5 {
		baseRadius = 0.5
	}
	if height < 0.75 {
		height = 0.75
	}

	return ConePars{
		Height:          height,
		BaseRadius:      baseRadius,
		TrianglesPerFan: trianglesPerFan,
	}
}

// ConeOverlaps checks if two cones overlap based on their base points and radii
func ConeOverlaps(base1, base2 Point2D, coneParams ConePars) bool {
	distance := Distance(base1, base2)
	radiusSum := coneParams.BaseRadius + coneParams.BaseRadius // Both cones use same parameters
	return distance < radiusSum
}
