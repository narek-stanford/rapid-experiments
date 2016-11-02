
public class LiazorPatker implements Patker {

	private IrakanPatker irakanPatker;
	private String fileName;

	public LiazorPatker(String fileName) {
		this.fileName = fileName;
	}

	@Override
	public void cucadrel() {
		if (irakanPatker == null) {
			irakanPatker = new IrakanPatker(fileName);
		}
		irakanPatker.cucadrel();
	}
}