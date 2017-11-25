package uk.ac.soton.ecs.bb4g15.utils;

import org.deeplearning4j.datasets.fetchers.BaseDataFetcher;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;

public class DLDataSetFetcher extends BaseDataFetcher {

	public DLDataSetFetcher(VFSGroupDataset<FImage> training) {
		
	}
	
	public DLDataSetFetcher(VFSListDataset<FImage> testing) {
		
	}
	
	@Override
	public void fetch(int numExamples) {
		// TODO Auto-generated method stub
		
	}
	
}
