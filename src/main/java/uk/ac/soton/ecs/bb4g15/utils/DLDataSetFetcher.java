package uk.ac.soton.ecs.bb4g15.utils;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;

import org.deeplearning4j.datasets.fetchers.BaseDataFetcher;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;

import uk.ac.soton.ecs.bb4g15.classifiers.Classifier;

public class DLDataSetFetcher extends BaseDataFetcher {

	public ArrayList<Entry<Integer, FImage>> images;
	public boolean train;

	public DLDataSetFetcher(VFSGroupDataset<FImage> training) {
		images = new ArrayList<Entry<Integer, FImage>>();
		for (String group : training.getGroups()) {
			for (int i = 0; i < training.getInstances(group).size(); i ++) {
				images.add(new AbstractMap.SimpleEntry<Integer, FImage>(getLabel(group), training.getInstances(group).get(i)));
			}
		}
		train = true;
		setup();
	}

	public DLDataSetFetcher(VFSListDataset<FImage> testing) {
		images = new ArrayList<Entry<Integer, FImage>>();
		for (int i = 0; i < testing.size(); i ++) {
			images.add(new AbstractMap.SimpleEntry<Integer, FImage>(0, testing.get(i)));
		}
		train = false;
		setup();
	}

	private void setup() {
		numOutcomes = 15;
		cursor = 0;
		inputColumns = DLDataSetIterator.IMAGE_DIMENSION * DLDataSetIterator.IMAGE_DIMENSION;
		totalExamples = images.size();
		reset();
	}

	public void fetch(int numExamples) {
		if (!hasMore()) {
			throw new IllegalStateException("Unable to getFromOrigin more; there are no more images");
		}

		float[][] featureData = new float[numExamples][0];
		float[][] labelData = new float[numExamples][0];

		int actualExamples = 0;

		Iterator<Entry<Integer, FImage>> it = images.iterator();
		int cur = 0;
		while (cur < cursor) {
			it.next();
			cur ++;
		}

		Entry<Integer, FImage> en;
		FImage processing;
		int newWidth = 0;
		int newHeight = 0;

		for (int i = 0; i < numExamples; i++, cursor++) {
			if (!hasMore()) { break; }

			en = it.next();

			processing = en.getValue();
			if (processing.getWidth() > processing.getHeight()) {
				newHeight = DLDataSetIterator.IMAGE_DIMENSION;
				newWidth = (int) Math.round(((double) processing.getWidth() / (double) processing.getHeight()) * (double) newHeight);
			} else {
				newWidth = DLDataSetIterator.IMAGE_DIMENSION;
				newHeight = (int) Math.round(((double) processing.getHeight() / (double) processing.getWidth()) * (double) newWidth);
			}

			processing = processing.process(new ResizeProcessor(newWidth, newHeight, false));
			processing = Classifier.crop(processing, DLDataSetIterator.IMAGE_DIMENSION, DLDataSetIterator.IMAGE_DIMENSION);

			byte[] img = processing.toByteImage();
			int label = en.getKey();

			float[] featureVec = new float[img.length];
			featureData[actualExamples] = featureVec;
			labelData[actualExamples] = new float[numOutcomes];
			labelData[actualExamples][label] = 1.0f;

			for (int j = 0; j < img.length; j++) {
				float v = ((int) img[j]) & 0xFF; //byte is loaded as signed -> convert to unsigned
				featureVec[j] = v / 255.0f;
			}
			actualExamples++;
		}

		if (actualExamples < numExamples) {
			featureData = Arrays.copyOfRange(featureData, 0, actualExamples);
			labelData = Arrays.copyOfRange(labelData, 0, actualExamples);
		}

		INDArray features = Nd4j.create(featureData);
		INDArray labels = Nd4j.create(labelData);
		curr = new DataSet(features, labels);
	}

	public void reset() {
		cursor = 0;
		curr = null;
		if (train) {
			Collections.shuffle(images);
		}
	}

	public DataSet next() {
		DataSet next = super.next();
		next.setLabelNames(DLDataSetIterator.getPredefinedLabels());
		return next;
	}

	/*High quality code incoming*/
	public static int getLabel(String group) {
		if (group.equals("bedroom")) { return 0; }	
		else if (group.equals("Coast")) { return 1; }
		else if (group.equals("Forest")) { return 2; }
		else if (group.equals("Highway")) { return 3; }
		else if (group.equals("industrial")) { return 4; }
		else if (group.equals("Insidecity")) { return 5; }
		else if (group.equals("kitchen")) { return 6; }
		else if (group.equals("livingroom")) { return 7; }
		else if (group.equals("Mountain")) { return 8; }
		else if (group.equals("Office")) { return 9; }
		else if (group.equals("OpenCountry")) { return 10; }
		else if (group.equals("store")) { return 11; }
		else if (group.equals("Street")) { return 12; }
		else if (group.equals("Suburb")) { return 13; }
		else { return 14; }
	}

	public static String getGroup(int label) {
		if (label == 0) { return "bedroom"; }
		else if (label == 1) { return "Coast"; }
		else if (label == 2) { return "Forest"; }
		else if (label == 3) { return "Highway"; }
		else if (label == 4) { return "industrial"; }
		else if (label == 5) { return "Insidecity"; }
		else if (label == 6) { return "kitchen"; }
		else if (label == 7) { return "livingroom"; }
		else if (label == 8) { return "Mountain"; }
		else if (label == 9) { return "Office"; }
		else if (label == 10) { return "OpenCountry"; }
		else if (label == 11) { return "store"; }
		else if (label == 12) { return "Street"; }
		else if (label == 13) { return "Suburb"; }
		else { return "TallBuilding"; }
	}
}