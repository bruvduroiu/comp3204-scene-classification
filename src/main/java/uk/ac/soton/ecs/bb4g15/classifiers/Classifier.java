package uk.ac.soton.ecs.bb4g15.classifiers;

import java.awt.Point;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.DoubleStream;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.MapBackedDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.image.FImage;

import uk.ac.soton.ecs.bb4g15.App;
import uk.ac.soton.ecs.bb4g15.utils.MListDataset;
import uk.ac.soton.ecs.bb4g15.utils.PreservedGroupedRandomSplitter;

public abstract class Classifier {
	protected GroupedDataset<String, MListDataset<FImage>, FImage> training;
	protected MListDataset<FImage> testing;
	

	public Classifier(VFSGroupDataset<FImage> training, VFSListDataset<FImage> testing) {
		this.training = new MapBackedDataset<String, MListDataset<FImage>, FImage>();
		for (String group : training.getGroups()) {
			((MapBackedDataset<String, MListDataset<FImage>, FImage>)(this.training)).add(group, new MListDataset<FImage>(training.get(group)));
		}

		this.testing = new MListDataset<FImage>(testing);
	}

	protected static DoubleFV normaliseVector(double[] vector) {
		double length = Math.sqrt(DoubleStream.of(vector)
				.map(i -> Math.pow(i, 2))
				.sum());

		vector = DoubleStream.of(vector)
				.map(i -> (i / length))
				.toArray();

		return new DoubleFV(vector);
	}

	protected static DoubleFV centerAndNormaliseVector(double[] vector) {
		double mean = DoubleStream.of(vector).average().getAsDouble();

		vector = DoubleStream.of(vector)
				.map(i -> (i - mean))
				.toArray();

		return normaliseVector(vector);
	}

	public Map<String, String> train() {
		GroupedDataset<String, MListDataset<FImage>, FImage> trainingBackup = training;
		MListDataset<FImage> testingBackup = testing;
		GroupedDataset<String, MListDataset<FImage>, FImage> testTemp;
		PreservedGroupedRandomSplitter<String, FImage> splits = 
				new PreservedGroupedRandomSplitter<String, FImage>(training, 80, 20);
		
		training = splits.getTrainingDataset();
		testTemp = splits.getTestDataset();

		testing = new MListDataset<FImage>();
		for (String group : testTemp.getGroups()) {
			for (int i = 0; i < testTemp.get(group).size(); i ++) {
				testing.add(testTemp.get(group).get(i), group + "/" + testTemp.get(group).getID(i));
			}
		}
		
		App.log("Testing on training split");
		Map<String, String> testres = _train(new HashMap<>());
		int correct = 0;
		int total = testing.size();
		for (Entry<String, String> result : testres.entrySet()) {
			boolean wasCorrect = result.getKey().split("/")[0].equals(result.getValue());
			correct += wasCorrect ? 1 : 0;
		}
		
		DecimalFormat df = new DecimalFormat("#.##");
		App.log("Classification accuracy: " + correct + "/" + total + " " + df.format(((double)correct / (double)total) * 100D) + "%");
		
		this.training = trainingBackup;
		this.testing = testingBackup;
		return _train(new HashMap<>());
	}

	protected abstract Map<String, String> _train(Map<String, String> classifications);

	public static FImage crop(FImage image, int cropWidth, int cropHeight) {
		Point center = new Point(image.getWidth()/2, image.getHeight()/2);

		float[][] crop = new float[cropHeight][cropWidth];

		int xStart = (center.x - (int) ((double)cropWidth/2D));
		int yStart = (center.y - (int) ((double)cropHeight/2D));

		for (int y = yStart; y < (center.y + (int) ((double)cropHeight/2D)); y++) {
			for (int x = xStart; x < (center.x + (int) ((double)cropWidth/2D)); x++) {
				int cropX = x - xStart;
				int cropY = y - yStart;
				crop[cropY][cropX] = image.pixels[y][x];
			}
		}
		image = image.internalAssign(new FImage(crop));
		return image;
	}
}