package scenario.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;

public class CSVReader {
	final private String separator;
	
	public CSVReader(String separator) {
		this.separator = separator;
	}
	
	public List<List<Integer>> load(File path, List<String> columns) throws IOException {
		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
		
		String line = null;
		List<String> header = null;
		List<String> row = null;
		
		List<List<Integer>> data = new LinkedList<>();
		
		while ((line = reader.readLine()) != null) {
			row = Arrays.asList(line.split(separator));
			
			if (header == null) {
				header = row;
			} else {
				List<Integer> dataRow = new ArrayList<>(columns.size());
				
				for (String column : columns) {
					dataRow.add(Integer.parseInt(row.get(header.indexOf(column))));
				}
				
				data.add(dataRow);
			}
		}		
		
		reader.close();
		
		return data;
	}
}
