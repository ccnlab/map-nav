// Copyright (c) 2019, The CCNLab Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package navenv

import (
	"github.com/emer/epe/epe"
	"github.com/goki/gi/mat32"
)

// MakeEmer constructs a new Emer virtual robot of given height (e.g., 1)
func MakeEmer(par *epe.Group, height float32) *epe.Group {
	emr := epe.AddNewGroup(par, "emer")
	width := height * .2
	body := epe.AddNewBox(emr, "body", mat32.Vec3{0, height / 2, 0}, mat32.Vec3{width, height, width})
	body.Mat.Color = "tan"
	headsz := width * .5
	head := epe.AddNewBox(emr, "head", mat32.Vec3{0, height/2 + headsz/2, 0}, mat32.Vec3{headsz, headz, headsz})
	head.Mat.Color = "tan"
	eyesz := headz * .1
	eyel := epe.AddNewBox(emr, "eye-l", mat32.Vec3{0, height/2 + headsz*.8, 0}, mat32.Vec3{headsz, headz, headsz})
	head.Mat.Color = "tan"
	return rm
}
