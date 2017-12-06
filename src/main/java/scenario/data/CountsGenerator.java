package scenario.data;

import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class CountsGenerator {
	public INDArray getCounts(List<List<Integer>> data) {
		return getCounts(data, data);
	}
	
	public INDArray getCounts(List<List<Integer>> data, List<List<Integer>> full) {
		int numberOfDimensions = data.get(0).size();
		int[] numberOfCategories = new int[numberOfDimensions];
		
		for (List<Integer> row : full) {
			for (int i = 0; i < numberOfDimensions; i++) {
				numberOfCategories[i] = Math.max(numberOfCategories[i], row.get(i) + 1);
			}
		}
		
		INDArray counts = Nd4j.zeros(numberOfCategories);
		
		for (List<Integer> row : data) {
			INDArrayIndex[] writeIndex = new INDArrayIndex[numberOfDimensions];
			int[] readIndex = new int[numberOfDimensions];
			
			for (int i = 0; i < numberOfDimensions; i++) {
				writeIndex[i] = NDArrayIndex.point(row.get(i));
				readIndex[i] = row.get(i);
			}
			
			counts.put(writeIndex, counts.getDouble(readIndex) + 1.0);
		}
		
		return counts;
	}
}
