package uk.ac.soton.ecs.bb4g15.utils;

import java.io.IOException;

import org.deeplearning4j.datasets.iterator.BaseDatasetIterator;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;

public class DLDataSetIterator extends BaseDatasetIterator {

	public DLDataSetIterator(int batch, VFSGroupDataset<FImage> training) throws IOException {
        super(batch, calculateTrainingSize(training), new DLDataSetFetcher(training));
    }
	
	public DLDataSetIterator(int batch, VFSListDataset<FImage> testing) {
		super(batch, calculateTestingSize(testing), new DLDataSetFetcher(testing));
	}

	public static int calculateTrainingSize(VFSGroupDataset<FImage> training) {
		return 0;
	}
	
	public static int calculateTestingSize(VFSListDataset<FImage> testing) {
		return 0;
	}
}