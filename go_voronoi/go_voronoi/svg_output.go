//
// Create an SVG output for Voronoi cells
//

package main

import (
	"fmt"
	"os"
)

// GenerateVoronoiSVG creates an SVG visualization of the Voronoi diagram
func GenerateVoronoiSVG(points []Point2D, grid [][]int, bounds [4]float64, gridSize int, filename string, svgWidth, svgHeight int) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create SVG file: %w", err)
	}
	defer file.Close()

	// Use provided SVG dimensions
	// SVG header
	fmt.Fprintf(file, `<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">`, svgWidth, svgHeight)

	// Calculate cell size
	cellWidth := float64(svgWidth) / float64(gridSize)
	cellHeight := float64(svgHeight) / float64(gridSize)

	// Draw grid cells with slight overlap to eliminate gaps
	for x := 0; x < gridSize; x++ {
		for y := 0; y < gridSize; y++ {
			dominantConeID := grid[x][y]
			if dominantConeID >= 0 && dominantConeID < len(points) {
				point := points[dominantConeID]
				svgX := float64(x) * cellWidth
				svgY := float64(y) * cellHeight
				// Add tiny overlap (0.1 pixel) to eliminate any gaps
				fmt.Fprintf(file, `<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" fill="rgb(%d,%d,%d)" stroke="none" shape-rendering="crispEdges" />`,
					svgX-0.05, svgY-0.05, cellWidth+0.1, cellHeight+0.1, 
					int(point.C.R*255), int(point.C.G*255), int(point.C.B*255))
			}
		}
	}

	// Optional: Draw original points (commented out for cleaner look)
	// for _, point := range points {
	// 	// Map world coordinates to SVG coordinates
	// 	svgX := (point.X - bounds[0]) / (bounds[2] - bounds[0]) * float64(svgWidth)
	// 	svgY := (point.Y - bounds[1]) / (bounds[3] - bounds[1]) * float64(svgHeight)
	// 	
	// 	fmt.Fprintf(file, `<circle cx="%.1f" cy="%.1f" r="8" fill="black" stroke="white" stroke-width="2" />`,
	// 		svgX, svgY)
	// }

	// Close SVG tag
	fmt.Fprintln(file, "</svg>")

	return nil
}

// GenerateVoronoiSVGDefault creates an SVG with default dimensions (800x800)
func GenerateVoronoiSVGDefault(points []Point2D, grid [][]int, bounds [4]float64, gridSize int, filename string) error {
	return GenerateVoronoiSVG(points, grid, bounds, gridSize, filename, 800, 800)
}