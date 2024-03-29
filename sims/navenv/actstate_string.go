// Code generated by "stringer -type=ActState"; DO NOT EDIT.

package navenv

import (
	"errors"
	"strconv"
)

var _ = errors.New("dummy error")

func _() {
	// An "invalid array index" compiler error signifies that the constant values have changed.
	// Re-run the stringer command to generate them again.
	var x [1]struct{}
	_ = x[NoActState-0]
	_ = x[MovingForward-1]
	_ = x[AvoidTurnHead-2]
	_ = x[ActStateN-3]
}

const _ActState_name = "NoActStateMovingForwardAvoidTurnHeadActStateN"

var _ActState_index = [...]uint8{0, 10, 23, 36, 45}

func (i ActState) String() string {
	if i < 0 || i >= ActState(len(_ActState_index)-1) {
		return "ActState(" + strconv.FormatInt(int64(i), 10) + ")"
	}
	return _ActState_name[_ActState_index[i]:_ActState_index[i+1]]
}

func (i *ActState) FromString(s string) error {
	for j := 0; j < len(_ActState_index)-1; j++ {
		if s == _ActState_name[_ActState_index[j]:_ActState_index[j+1]] {
			*i = ActState(j)
			return nil
		}
	}
	return errors.New("String: " + s + " is not a valid option for type: ActState")
}
