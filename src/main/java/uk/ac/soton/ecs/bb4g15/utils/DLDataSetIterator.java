package uk.ac.soton.ecs.bb4g15.utils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.datasets.iterator.BaseDatasetIterator;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;

public class DLDataSetIterator extends BaseDatasetIterator {

	public static final int IMAGE_DIMENSION = 64;
	
	public DLDataSetIterator(int batch, VFSGroupDataset<FImage> training) throws IOException {
        super(batch, calculateTrainingSize(training), new DLDataSetFetcher(training));
    }
	
	public DLDataSetIterator(int batch, VFSListDataset<FImage> testing) {
		super(batch, calculateTestingSize(testing), new DLDataSetFetcher(testing));
	}

	private static int calculateTrainingSize(VFSGroupDataset<FImage> training) {
		int size = 0;
		for (String group : training.getGroups()) {
			size += training.get(group).size();
		}
		return size;
	}
	
	private static int calculateTestingSize(VFSListDataset<FImage> testing) {
		return testing.size();
	}
	
	public static List<String> getPredefinedLabels() {
		List<String> labels = new ArrayList<String>();
		labels.add("bedroom");
		labels.add("Coast");
		labels.add("Forest");
		labels.add("Highway");
		labels.add("industrial");
		labels.add("InsideCity");
		labels.add("kitchen");
		labels.add("livingroom");
		labels.add("Mountain");
		labels.add("Office");
		labels.add("OpenCountry");
		labels.add("store");
		labels.add("Street");
		labels.add("Suburb");
		labels.add("TallBuilding");
		return labels;
	}
	
	public List<String> getLabels() {
        return getPredefinedLabels();
    }
}