package uk.ac.soton.ecs.bb4g15.classifiers;

import java.util.Map;

import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;

public class Custom extends Classifier {

	public Custom(VFSGroupDataset<FImage> training, VFSListDataset<FImage> testing) {
		super(training, testing);
	}

	@Override
	protected Map<String, String> _train(Map<String, String> classifications) {
		return classifications;
	}
}
