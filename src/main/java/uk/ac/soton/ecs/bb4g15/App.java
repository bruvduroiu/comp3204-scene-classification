package uk.ac.soton.ecs.bb4g15;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.Map;

import org.apache.commons.vfs2.FileSystemException;

import uk.ac.soton.ecs.bb4g15.classifiers.Classifier;
import uk.ac.soton.ecs.bb4g15.classifiers.Custom;
import uk.ac.soton.ecs.bb4g15.classifiers.KNN;
import uk.ac.soton.ecs.bb4g15.classifiers.Linear;
import uk.ac.soton.ecs.bb4g15.utils.Dataset;


public class App {
	private static final boolean LOG = true;
	public static void main( String[] args ) {
		try {
			Classifier classifier;
			classifier = new KNN(Dataset.loadTraining(), Dataset.loadTesting(), 3);
			outputPredictions(1, classifier.train());
			classifier = new Linear(Dataset.loadTraining(), Dataset.loadTesting(), 8, 4, 500);
			outputPredictions(2, classifier.train());
			classifier = new Custom(Dataset.loadTraining(), Dataset.loadTesting());
			outputPredictions(3, classifier.train());

		} catch (FileSystemException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void log(String output) {
		if (LOG) { System.out.println(output); }
	}
	
	public static void outputPredictions(int runNum, Map<String, String> results) throws Exception {
		App.log("Outputting predictions for run #" + runNum);
		File f = new File("run" + runNum + ".txt");
		BufferedWriter bw = new BufferedWriter(new FileWriter(f));
		for(Map.Entry<String, String> classification : results.entrySet()) {
			bw.write(classification.getKey() + " " + classification.getValue());
			bw.newLine();
		}
		bw.close();
	}
}
