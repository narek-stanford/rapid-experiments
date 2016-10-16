
public class IrakanPatker implements Patker {

	private String fileName;

	public IrakanPatker(String fileName) {
		this.fileName = fileName;
		loadFromDisk(fileName);
	}

	@Override
	public void cucadrel() {
		System.out.println("Cucadrum em " + fileName);
	}
}