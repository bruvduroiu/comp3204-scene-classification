package uk.ac.soton.ecs.bb4g15.classifiers;

import java.awt.Point;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.DoubleStream;

import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;

public abstract class Classifier {
	protected VFSGroupDataset<FImage> training;
	protected VFSListDataset<FImage> testing;

	public Classifier(VFSGroupDataset<FImage> training, VFSListDataset<FImage> testing) {
		this.training = training;
		this.testing = testing;
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

	public Map<String, String> train() { return _train(new HashMap<>()); }

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