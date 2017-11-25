package uk.ac.soton.ecs.bb4g15.classifiers;

import java.awt.Point;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.DoubleCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.DoubleKMeans;
import org.openimaj.util.pair.IntDoublePair;

import de.bwaldvogel.liblinear.SolverType;
import uk.ac.soton.ecs.bb4g15.App;

public class Linear extends Classifier {

	private int patchSize;
	private int sampleEvery;
	private int quantiserTrainSize;

	public Linear(VFSGroupDataset<FImage> training, VFSListDataset<FImage> testing, int patchSize, int sampleEvery, int quantiserTrainSize) {
		super(training, testing);
		this.patchSize = patchSize;
		this.sampleEvery = sampleEvery;
		this.quantiserTrainSize = quantiserTrainSize;
	}

	protected Map<String, String> _train(Map<String, String> classifications) {
		App.log("Training quantiser");

		//Trains the quantiser with patches from 100 images
		HardAssigner<double[], double[], IntDoublePair> assigner = trainQuantiser(GroupedUniformRandomisedSampler.sample(training, 100));

		//Extracts dense features from the images
		DenseExtractor extractor = new DenseExtractor(assigner);

		LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<FImage, String>(
				extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);

		App.log("Training annotator");
		ann.train(training);

		App.log("Testing annotator");
		for (int i = 0; i < testing.size(); i++) {
			String prediction = (String) ann.classify(testing.get(i)).getPredictedClasses().toArray()[0];
			classifications.put(testing.getID(i), prediction);
		}

		return classifications;
	}

	/*
	 * Starts in the top left corner and places a patch of a specified size of pixels and transforms that into a vector
	 * Moves this patch across and then down
	 */
	private FImage[][] getPatches(FImage image) {
		FImage[][] patches = new FImage[(int)Math.floor((image.getWidth() - patchSize) / sampleEvery) + 1][(int)Math.floor((image.getHeight() - patchSize) / sampleEvery) + 1];
		for (int i = 0; i < patches.length; i ++) {
			for (int i2 = 0; i2 < patches[i].length; i2 ++) {
				Point bottomRight = new Point(i * sampleEvery + patchSize, i2 * sampleEvery + patchSize);

				float[][] crop = new float[patchSize][patchSize]; //the crop of the patch

				//fills in the crop
				for (int y = bottomRight.y - patchSize; y < bottomRight.y; y++) {
					for (int x = bottomRight.x - patchSize; x < bottomRight.x; x++) {
						int cropX = x - (bottomRight.x - patchSize);
						int cropY = y - (bottomRight.y - patchSize);

						crop[cropY][cropX] = image.pixels[y][x];
					}
				}
				patches[i][i2] = new FImage(crop);
			}
		}
		return patches;
	}

	//Helper method converting a double[] to a Double[]
	private Double[] convert(double[] in) {
		Double[] ret = new Double[in.length];
		for (int i = 0; i < in.length; i ++) {
			ret[i] = in[i];
		}
		return ret;
	}

	//Helper method converting a Double[] to a double[]
	private double[] convert(Double[] in) {
		double[] ret = new double[in.length];
		for (int i = 0; i < in.length; i ++) {
			ret[i] = in[i];
		}
		return ret;
	}

	//Transforms a list of Double[] to a 2d double array
	private double[][] fromList(List<Double[]> list) {
		double[][] ret = new double[list.size()][];
		for (int i = 0; i < list.size(); i ++) {
			ret[i] = convert(list.get(i));
		}
		return ret;
	}

	HardAssigner<double[], double[], IntDoublePair> trainQuantiser (
			Dataset<FImage> sample)
	{
		List<Double[]> allkeys = new ArrayList<Double[]>(); //array of all the patch vectors

		FImage[][] patches;

		for (FImage img : sample) {
			patches = getPatches(img);
			for (int i = 0; i < patches.length; i ++) {
				for (int i2 = 0; i2 < patches[i].length; i2 ++) {
					allkeys.add(convert(centerAndNormaliseVector(patches[i][i2].getDoublePixelVector()).asDoubleVector()));
				}
			}
		}

		Collections.shuffle(allkeys); //shuffles the patches so it isn't just training using patches from 1 image

		if (allkeys.size() > 10000)
			allkeys = allkeys.subList(0, 10000);

		//Using KMeans as specified
		DoubleKMeans km = DoubleKMeans.createKDTreeEnsemble(quantiserTrainSize);
		DoubleCentroidsResult result = km.cluster(fromList(allkeys));

		return result.defaultHardAssigner();
	}

	class DenseExtractor implements FeatureExtractor<DoubleFV, FImage> {
		HardAssigner<double[], double[], IntDoublePair> assigner;

		public DenseExtractor(HardAssigner<double[], double[], IntDoublePair> assigner)
		{
			this.assigner = assigner;
		}

		//Maps the patches to a visual word
		public DoubleFV extractFeature(FImage image) {
			FImage[][] patches = getPatches(image);
			BagOfVisualWords<double[]> bovw = new BagOfVisualWords<double[]>(assigner);

			List<double[]> features = new ArrayList<double[]>(); //list of all the patch vectors
			for (int i = 0; i < patches.length; i ++) {
				for (int i2 = 0; i2 < patches[i].length; i2 ++) {
					features.add(centerAndNormaliseVector(patches[i][i2].getDoublePixelVector()).asDoubleVector());
				}
			}
			
			return bovw.aggregateVectorsRaw(features).asDoubleFV();
		}
	}
}