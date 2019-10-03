// Copyright (c) 2019, The CCNLab Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package navenv

import (
	"github.com/emer/eve/eve"
	"github.com/goki/gi/mat32"
)

// RoomParams
type RoomParams struct {
	Width  float32 `desc:"width of room"`
	Depth  float32 `desc:"depth of room"`
	Height float32 `desc:"height of room"`
	Thick  float32 `desc:"thickness of walls of room"`
}

func (rp *RoomParams) Defaults() {
	rp.Width = 10
	rp.Depth = 15
	rp.Height = 2
	rp.Thick = 0.2
}

// MakeRoom constructs a new room in given parent group with given params
func (rp *RoomParams) MakeRoom(par *eve.Group, name string) *eve.Group {
	rm := eve.AddNewGroup(par, name)
	floor := eve.AddNewBox(rm, "floor", mat32.Vec3{0, -rp.Thick / 2, 0}, mat32.Vec3{rp.Width, rp.Thick, rp.Depth})
	floor.Mat.Color = "grey"
	bwall := eve.AddNewBox(rm, "back-wall", mat32.Vec3{0, rp.Height / 2, -rp.Depth / 2}, mat32.Vec3{rp.Width, rp.Height, rp.Thick})
	bwall.Mat.Color = "blue"
	lwall := eve.AddNewBox(rm, "left-wall", mat32.Vec3{-rp.Width / 2, rp.Height / 2, 0}, mat32.Vec3{rp.Thick, rp.Height, rp.Depth})
	lwall.Mat.Color = "red"
	rwall := eve.AddNewBox(rm, "right-wall", mat32.Vec3{rp.Width / 2, rp.Height / 2, 0}, mat32.Vec3{rp.Thick, rp.Height, rp.Depth})
	rwall.Mat.Color = "green"
	fwall := eve.AddNewBox(rm, "front-wall", mat32.Vec3{0, rp.Height / 2, rp.Depth / 2}, mat32.Vec3{rp.Width, rp.Height, rp.Thick})
	fwall.Mat.Color = "yellow"
	return rm
}
