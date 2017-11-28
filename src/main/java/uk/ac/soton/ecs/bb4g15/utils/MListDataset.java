package uk.ac.soton.ecs.bb4g15.utils;

import java.util.ArrayList;

import org.openimaj.data.dataset.ListBackedDataset;
import org.openimaj.data.dataset.VFSListDataset;

//Adds file names to a generic list data set
public class MListDataset<T> extends ListBackedDataset<T> {

	public ArrayList<String> ids;

	public MListDataset() {
		super();
		ids = new ArrayList<String>();
	}
	
	public MListDataset (ListBackedDataset<T> input) {
		super();
		ids = new ArrayList<String>();
		for (T in : input) {
			add(in);
		}
	}
	
	public MListDataset (VFSListDataset<T> input) {
		super();

		ids = new ArrayList<String>();
		for (int i = 0; i < input.size(); i ++) {
			ids.add(input.getID(i));
			super.add(input.get(i));
		}
	}

	public boolean add(T in) {
		return add(in, "");
	}
	
	public boolean add(T in, String id) {
		ids.add(id);
		return super.add(in);
	}
	
	public String getID(int i) { return ids.get(i); }
}