// Copyright (c) 2019, The CCNLab Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package navenv

import (
	"github.com/emer/epe/epe"
	"github.com/goki/gi/mat32"
)

// MakeRoom constructs a new room in given parent group with given params
func MakeRoom(par *epe.Group, name string, width, depth, height, thick float32) *epe.Group {
	rm := epe.AddNewGroup(par, name)
	bwall := epe.AddNewBox(rm, "back-wall", mat32.Vec3{0, height / 2, -depth / 2}, mat32.Vec3{width, height, thick})
	bwall.Mat.Color = "tan"
	lwall := epe.AddNewBox(rm, "left-wall", mat32.Vec3{-width / 2, height / 2, 0}, mat32.Vec3{thick, height, depth})
	lwall.Mat.Color = "red"
	rwall := epe.AddNewBox(rm, "right-wall", mat32.Vec3{width / 2, height / 2, 0}, mat32.Vec3{thick, height, depth})
	rwall.Mat.Color = "green"
	fwall := epe.AddNewBox(rm, "front-wall", mat32.Vec3{0, height / 2, depth / 2}, mat32.Vec3{width, height, thick})
	bwall.Mat.Color = "yellow"
	return rm
}
