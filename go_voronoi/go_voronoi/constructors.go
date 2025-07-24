//
// Constructors for Voronoi diagram types
//

package main

// NewPoint2D creates a new Point2D with the given coordinates and color
func NewPoint2D(x, y, r, g, b float64) Point2D {
	return Point2D{X: x, Y: y, C: Color{R: r, G: g, B: b}}
}

// NewPoint3D creates a new Point3D with the given coordinates
func NewPoint3D(x, y, z float64) Point3D {
	return Point3D{X: x, Y: y, Z: z}
}

// NewTriangle creates a new Triangle with the given vertices and color
func NewTriangle(v1, v2, v3 Point3D, color Color) Triangle {
	return Triangle{V1: v1, V2: v2, V3: v3, Color: color}
}
 
// NewVoronoiCell creates a new VoronoiCell with the given vertices and color
func NewVoronoiCell(vertices []Point2D, color Color) VoronoiCell {
	return VoronoiCell{Vertices: vertices, Color: color}
}

// NewConePars creates a new ConePars with the given parameters
func NewConePars(height, baseRadius float64, trianglesPerFan int) ConePars {
	return ConePars{Height: height, BaseRadius: baseRadius, TrianglesPerFan: trianglesPerFan}
}

// ConeParsValid checks if the ConePars parameters are valid
func ConeParsValid(cone ConePars) bool {
	return cone.Height > 0 && cone.BaseRadius > 0 && cone.TrianglesPerFan >= 3
}
