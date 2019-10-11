// Copyright (c) 2019, The CCNLab Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package navenv

// CurPrvVel is basic state management for current, previous and velocity (diff)
type CurPrvVel struct {
	Cur float32 `desc:"current value"`
	Prv float32 `desc:"previous value"`
	Vel float32 `desc:"velocity as difference: Cur - Prv"`
}

// Update updates the new current value, copying Cur to Prv and computing Vel
func (cv *CurPrvVel) Update(cur float32) {
	cv.Prv = cv.Cur
	cv.Cur = cur
	cv.Vel = cv.Cur - cv.Prv
}
