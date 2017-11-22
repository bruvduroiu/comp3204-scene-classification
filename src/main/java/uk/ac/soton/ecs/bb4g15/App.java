package uk.ac.soton.ecs.bb4g15;

import org.apache.commons.vfs2.FileSystemException;
import uk.ac.soton.ecs.bb4g15.classifiers.KNN;
import uk.ac.soton.ecs.bb4g15.utils.Dataset;

import java.util.Map;


public class App {

    public static void main( String[] args ) {
        try {
            KNN classifier = new KNN(Dataset.loadTraining(), Dataset.loadTesting());

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
}
