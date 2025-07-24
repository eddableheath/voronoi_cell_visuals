//
// Extracts Voronoi boundaries from a set of points and their interesting projected cones
//

package main

// GetConeHeightAtPoint calculates the height of a cone at a given 2D point
func GetConeHeightAtPoint(queryPoint Point2D, coneBase Point2D, coneParams ConePars) float64 {
	// Calculate the distance from the query point to the cone base
	distance := Distance(queryPoint, coneBase)

	// If the distance is greater than the base radius, height is 0
	if distance > coneParams.BaseRadius {
		return 0.0
	}

	// Calculate the height based on the distance from the base
	height := coneParams.Height * (1 - (distance / coneParams.BaseRadius))
	return height
}

// FindDominantCone determines which cone is highest at a given point
func FindDominantCone(queryPoint Point2D, points []Point2D, coneParams ConePars) (int, float64) {
	maxHeight := 0.0
	dominantConeID := -1

	for i, basePoint := range points {
		height := GetConeHeightAtPoint(queryPoint, basePoint, coneParams)
		if height > maxHeight {
			maxHeight = height
			dominantConeID = i
		}
	}

	return dominantConeID, maxHeight
}

// SampleVoronoiGrid creates a grid and determines dominant cone for each point
func SampleVoronoiGrid(points []Point2D, coneParams ConePars, gridSize int, bounds [4]float64) [][]int {
	grid := make([][]int, gridSize)
	for i := range grid {
		grid[i] = make([]int, gridSize)
	}

	for x := 0; x < gridSize; x++ {
		for y := 0; y < gridSize; y++ {
			queryPoint := Point2D{
				X: bounds[0] + (bounds[2]-bounds[0])*float64(x)/float64(gridSize-1),
				Y: bounds[1] + (bounds[3]-bounds[1])*float64(y)/float64(gridSize-1),
			}

			dominantConeID, _ := FindDominantCone(queryPoint, points, coneParams)
			grid[x][y] = dominantConeID
		}
	}

	return grid
}