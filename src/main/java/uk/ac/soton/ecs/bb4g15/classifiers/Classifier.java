package uk.ac.soton.ecs.bb4g15.classifiers;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.DoubleStream;

import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.image.FImage;

public abstract class Classifier {
    protected VFSGroupDataset<FImage> training;
    protected VFSListDataset<FImage> testing;
  
	public Classifier(VFSGroupDataset<FImage> training, VFSListDataset<FImage> testing) {
		this.training = training;
		this.testing = testing;
	}
	
	protected static DoubleFV normaliseVector(double[] vector) {
        double mean = DoubleStream.of(vector).average().getAsDouble();
        double length = Math.sqrt(DoubleStream.of(vector)
                .map(i -> Math.pow(i, 2))
                .sum());
        vector = DoubleStream.of(vector)
                .map(i -> (i / length))
                .toArray();
        //vector = DoubleStream.of(vector)
        //        .map(i -> (i / mean) / length)
        //        .toArray();
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
		return _train(new HashMap<>());
	}
	
	protected abstract Map<String, String> _train(Map<String, String> classifications);
}
