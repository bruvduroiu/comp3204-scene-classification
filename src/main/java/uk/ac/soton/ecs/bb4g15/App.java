package uk.ac.soton.ecs.bb4g15;

import org.apache.commons.vfs2.FileSystemException;

import uk.ac.soton.ecs.bb4g15.classifiers.Classifier;
import uk.ac.soton.ecs.bb4g15.classifiers.KNN;
import uk.ac.soton.ecs.bb4g15.classifiers.Linear;
import uk.ac.soton.ecs.bb4g15.utils.Dataset;

import java.util.Map;


public class App {
	private static final boolean LOG = true;
	public static void main( String[] args ) {
		try {
			Classifier classifier;
			classifier = new KNN(Dataset.loadTraining(), Dataset.loadTesting(), 3);
			classifier = new Linear(Dataset.loadTraining(), Dataset.loadTesting(), 8, 4, 500);

			Map<String, String> results = classifier.train();

			for(Map.Entry<String, String> classification : results.entrySet()) {
				String file = classification.getKey();
				String result = classification.getValue();
				System.out.println(file + ": " + result);
			}

		} catch (FileSystemException e) {
			e.printStackTrace();
		}
	}
	
	public static void log(String output) {
		if (LOG) { System.out.println(output); }
	}
}
