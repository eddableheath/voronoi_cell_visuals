//
// Cone constructor for Voronoi cells
//

package main

import (
	"math"
)

// GenerateCone creates a cone structure based on the base point and parameters
func GenerateCone(base Point2D, height, baseRadius float64, trianglesPerFan int) []Triangle {
	cone := make([]Triangle, trianglesPerFan)
	color := base.C // Use the color from the base point

	// Calculate angle step for the cone fan
	angleStep := 2 * math.Pi / float64(trianglesPerFan)

	// Base center and apex points
	baseCenter := Point3D{X: base.X, Y: base.Y, Z: 0}
	apex := Point3D{X: base.X, Y: base.Y, Z: height}

	for i := 0; i < trianglesPerFan; i++ {
		// Current angle for the base perimeter
		currentAngle := float64(i) * angleStep

		// Base perimeter point
		currentBase := Point3D{
			X: base.X + baseRadius*math.Cos(currentAngle),
			Y: base.Y + baseRadius*math.Sin(currentAngle),
			Z: 0,
		}

		// Create triangle: base center -> current base -> apex
		cone[i] = NewTriangle(baseCenter, currentBase, apex, color)
	}

	return cone
}

// ConstructCone generates a cone structure given a base point and parameters
func ConstructCone(coneParams ConePars, base Point2D) []Triangle {
	cone := GenerateCone(base, coneParams.Height, coneParams.BaseRadius, coneParams.TrianglesPerFan)
	
	// The cone is already properly constructed with the right heights and colors
	// No need to modify the triangles here
	return cone
}