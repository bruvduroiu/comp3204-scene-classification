package uk.ac.soton.ecs.bb4g15.classifiers;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;

import uk.ac.soton.ecs.bb4g15.App;
import uk.ac.soton.ecs.bb4g15.utils.DLDataSetIterator;

public class Custom extends Classifier {

	public DLDataSetIterator trainIterator;
	public DLDataSetIterator testIterator;
	public static final int batchSize = 64; // Test batch size


	public Custom(VFSGroupDataset<FImage> training, VFSListDataset<FImage> testing) throws Exception {
		super(training, testing);
		this.trainIterator = new DLDataSetIterator(batchSize, training);
		this.testIterator = new DLDataSetIterator(batchSize, testing);
	}

	@Override
	protected Map<String, String> _train(Map<String, String> classifications) {
		int nEpochs = 5000; // Number of training epochs

		MultiLayerNetwork model = new MultiLayerNetwork(getNetConfig());
		model.init();

		model.setListeners(new ScoreIterationListener(1));
		for (int i = 0; i < nEpochs; i ++) {
			model.fit(trainIterator);
			App.log("Completed epoch " + i);
		}

		DataSet test;
		while (testIterator.hasNext()) {
			test = testIterator.next();
			classifications.put("", model.predict(test).get(0));
		}

		return classifications;
	}

	public MultiLayerConfiguration getNetConfig() {
		int nChannels = 1; // Number of input channels
		int outputNum = 15; // The number of possible outcomes

		Map<Integer, Double> lrSchedule = new HashMap<>();
		lrSchedule.put(0, 0.01);
		lrSchedule.put(1000, 0.005);
		lrSchedule.put(3000, 0.001);

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.iterations(1)
				.regularization(true).l2(0.0005)
				.learningRate(.01)
				.biasLearningRate(0.02)
				.learningRateDecayPolicy(LearningRatePolicy.Schedule)
				.learningRateDecayPolicy(LearningRatePolicy.Inverse)
				.lrPolicyDecayRate(0.001)
				.lrPolicyPower(0.75)
				.weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.dropOut(0.55)
				.updater(new Nesterovs(0.75))
				.list()
				.layer(0, new ConvolutionLayer.Builder(5, 5)
						//nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
						.nIn(nChannels)
						.stride(1, 1)
						.nOut(20)
						.activation(Activation.IDENTITY)
						.build())
				.layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2,2)
						.stride(2,2)
						.build())
				.layer(2, new ConvolutionLayer.Builder(5, 5)
						.stride(1, 1)
						.nOut(50)
						.activation(Activation.IDENTITY)
						.build())
				.layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2,2)
						.stride(2,2)
						.build())
				.layer(4, new DenseLayer.Builder().activation(Activation.RELU)
						.nOut(500).build())
				.layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.nOut(outputNum)
						.activation(Activation.SOFTMAX)
						.build())
				.setInputType(InputType.convolutionalFlat(28,28,1))
				.backprop(true).pretrain(false).build();
		return conf;
	}
}