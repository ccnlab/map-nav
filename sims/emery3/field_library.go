package main

import (
	"github.com/emer/etable/etensor"
	"github.com/goki/gi/gi"
	"path/filepath"
	"strings"
)

func (ss *Sim) ConfigRFMaps() { // TODO(refactor): environment file
	ss.RFMaps = make(map[string]*etensor.Float32)
	mt := &etensor.Float32{}
	mt.CopyShapeFrom(ss.TrainEnv.World)
	ss.RFMaps["Pos"] = mt

	mt = &etensor.Float32{}
	mt.SetShape([]int{len(ss.TrainEnv.Acts)}, nil, nil)
	ss.RFMaps["Act"] = mt

	mt = &etensor.Float32{}
	mt.SetShape([]int{ss.TrainEnv.NRotAngles}, nil, nil)
	ss.RFMaps["Ang"] = mt

	mt = &etensor.Float32{}
	mt.SetShape([]int{3}, nil, nil)
	ss.RFMaps["Rot"] = mt
}

// SetAFMetaData
func (ss *Sim) SetAFMetaData(af etensor.Tensor) { // TODO(refactor): game gui related
	af.SetMetaData("min", "0")
	af.SetMetaData("colormap", "Viridis") // "JetMuted")
	af.SetMetaData("grid-fill", "1")
}

// UpdtARFs updates position activation rf's
func (ss *Sim) UpdtARFs() { // TODO(refactor): game gui
	for nm, mt := range ss.RFMaps {
		mt.SetZeros()
		switch nm {
		case "Pos":
			mt.Set([]int{ss.TrainEnv.PosI.Y, ss.TrainEnv.PosI.X}, 1)
		case "Act":
			mt.Set1D(ss.TrainEnv.Act, 1)
		case "Ang":
			mt.Set1D(ss.TrainEnv.Angle/15, 1)
		case "Rot":
			mt.Set1D(1+ss.TrainEnv.RotAng/15, 1)
		}
	}

	naf := len(ss.ARFLayers) * len(ss.RFMaps)
	if len(ss.ARFs.RFs) != naf {
		for _, lnm := range ss.ARFLayers {
			ly := ss.Net.LayerByName(lnm)
			if ly == nil {
				continue
			}
			vt := ss.ValsTsr(lnm)
			ly.UnitValsTensor(vt, "ActM")
			for nm, mt := range ss.RFMaps {
				af := ss.ARFs.AddRF(lnm+"_"+nm, vt, mt)
				ss.SetAFMetaData(&af.NormRF)
			}
		}
	}
	for _, lnm := range ss.ARFLayers {
		ly := ss.Net.LayerByName(lnm)
		if ly == nil {
			continue
		}
		vt := ss.ValsTsr(lnm)
		ly.UnitValsTensor(vt, "ActM")
		for nm, mt := range ss.RFMaps {
			ss.ARFs.Add(lnm+"_"+nm, vt, mt, 0.01) // thr prevent weird artifacts
		}
	}
}

// SaveAllARFs saves all ARFs to files
func (ss *Sim) SaveAllARFs() { // TODO(refactor): game gui
	ss.ARFs.Avg()
	ss.ARFs.Norm()
	//for _, paf := range ss.ARFs.RFs { // TODO Deleted logging
	//	fnm := ss.LogFileName(paf.Name)
	//	etensor.SaveCSV(&paf.NormRF, gi.FileName(fnm), '\t')
	//}
}

// OpenAllARFs open all ARFs from directory of given path
func (ss *Sim) OpenAllARFs(path gi.FileName) { // TODO(refactor): game gui
	ss.UpdtARFs()
	ss.ARFs.Avg()
	ss.ARFs.Norm()
	ap := string(path)
	if strings.HasSuffix(ap, ".tsv") {
		ap, _ = filepath.Split(ap)
	}
	//vp := ss.Win.Viewport // TODO Deleted logging
	//for _, paf := range ss.ARFs.RFs {
	//	fnm := filepath.Join(ap, ss.LogFileName(paf.Name))
	//	err := etensor.OpenCSV(&paf.NormRF, gi.FileName(fnm), '\t')
	//	if err != nil {
	//		fmt.Println(err)
	//	} else {
	//		etview.TensorGridDialog(vp, &paf.NormRF, giv.DlgOpts{Title: "Act RF " + paf.Name, Prompt: paf.Name, TmpSave: nil}, nil, nil)
	//	}
	//}
}
