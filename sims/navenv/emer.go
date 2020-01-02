// Copyright (c) 2019, The CCNLab Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package navenv

import (
	"github.com/emer/eve/eve"
	"github.com/goki/gi/mat32"
)

// MakeEmer constructs a new Emer virtual robot of given height (e.g., 1)
func MakeEmer(par *eve.Group, height float32) *eve.Group {
	emr := eve.AddNewGroup(par, "emer")
	width := height * .4
	depth := height * .15
	body := eve.AddNewBox(emr, "body", mat32.Vec3{0, height / 2, 0}, mat32.Vec3{width, height, depth})
	body.Color = "purple"

	headsz := depth * 1.5
	hhsz := .5 * headsz
	hgp := eve.AddNewGroup(emr, "head")
	hgp.Initial.Pos = mat32.Vec3{0, height + hhsz, 0}

	head := eve.AddNewBox(hgp, "head", mat32.Vec3{0, 0, 0}, mat32.Vec3{headsz, headsz, headsz})
	head.Color = "tan"
	eyesz := headsz * .2
	eyel := eve.AddNewBox(hgp, "eye-l", mat32.Vec3{-hhsz * .6, headsz * .1, -(hhsz + eyesz*.3)}, mat32.Vec3{eyesz, eyesz * .5, eyesz * .2})
	eyel.Color = "green"
	eyer := eve.AddNewBox(hgp, "eye-r", mat32.Vec3{hhsz * .6, headsz * .1, -(hhsz + eyesz*.3)}, mat32.Vec3{eyesz, eyesz * .5, eyesz * .2})
	eyer.Color = "green"
	return emr
}
