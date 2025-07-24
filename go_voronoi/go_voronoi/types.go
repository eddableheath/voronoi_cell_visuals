//
// Package govoronoi provides types and functions for Voronoi diagram calculations.
//

package main

// Color represents RGB color
type Color struct {
	R, G, B float64 // RGB values [0,1]
}

// Point2D represets a 2D point with RGB color
type Point2D struct {
	X, Y float64
	C    Color // Color of the point
}

// Point3D represents a 3D point
type Point3D struct {
	X, Y, Z float64
}

// Triangle represents a triangle in 3D space
type Triangle struct {
	V1, V2, V3 Point3D
	Color       Color // Color of the triangle
}

// ConePars implements parameters for cone generation
type ConePars struct {
	BasePoint      Point2D // Base point of the cone
	Height         float64 // Height of the cones
	BaseRadius     float64 // Base radius of the cones
	TrianglesPerFan int    // Number of triangles in each cone fan
}

// VoronoiCell represents a Voronoi cell with its boundary and color
type VoronoiCell struct {
	Vertices []Point2D // Vertices of the cell boundary
	Color    Color      // Color of the cell
}
