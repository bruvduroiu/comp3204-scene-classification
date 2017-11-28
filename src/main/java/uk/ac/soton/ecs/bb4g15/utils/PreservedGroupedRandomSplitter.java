package uk.ac.soton.ecs.bb4g15.utils;

import java.util.Map.Entry;

import org.openimaj.data.RandomData;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.MapBackedDataset;

//Makes sure that MListDataset is returned instead of a ListDataset
//Couldn't override it nicely so had to rewrite it... :(
public class PreservedGroupedRandomSplitter<KEY, INSTANCE> {

	private int numTraining;
	private int numTesting;
	private GroupedDataset<KEY, MListDataset<INSTANCE>, INSTANCE> trainingSplit;
	private GroupedDataset<KEY, MListDataset<INSTANCE>, INSTANCE> testingSplit;
	private GroupedDataset<KEY, ? extends MListDataset<INSTANCE>, INSTANCE> dataset;
	
	public PreservedGroupedRandomSplitter(GroupedDataset<KEY, MListDataset<INSTANCE>, INSTANCE> dataset, int numTraining, int numTesting) {
		this.dataset = dataset;
		this.numTraining = numTraining;
		this.numTesting = numTesting;
		recomputeSubsets();
	}
	
	public void recomputeSubsets() {
		trainingSplit = new MapBackedDataset<KEY, MListDataset<INSTANCE>, INSTANCE>();
		testingSplit = new MapBackedDataset<KEY, MListDataset<INSTANCE>, INSTANCE>();

		for (final Entry<KEY, ? extends MListDataset<INSTANCE>> e : dataset.entrySet()) {
			final KEY key = e.getKey();
			final MListDataset<INSTANCE> allData = e.getValue();

			if (allData.size() < numTraining + 1)
				throw new RuntimeException(
						"Too many training examples; none would be available for validation or testing.");

			final int[] ids = RandomData.getUniqueRandomInts(
					Math.min(numTraining + numTesting, allData.size()), 0,
					allData.size());

			final MListDataset<INSTANCE> train = new MListDataset<INSTANCE>();
			for (int i = 0; i < numTraining; i++) {
				train.add(allData.get(ids[i]), allData.getID(i));
			}
			trainingSplit.put(key, train);

			final MListDataset<INSTANCE> test = new MListDataset<INSTANCE>();
			for (int i = numTraining; i < ids.length; i++) {
				test.add(allData.get(ids[i]), allData.getID(i));
			}
			testingSplit.put(key, test);
		}
	}
	
	public GroupedDataset<KEY, MListDataset<INSTANCE>, INSTANCE> getTestDataset() {
		return testingSplit;
	}

	public GroupedDataset<KEY, MListDataset<INSTANCE>, INSTANCE> getTrainingDataset() {
		return trainingSplit;
	}
}