package rutgers.hwsel.ProtectivePerturbation;

import android.annotation.SuppressLint;
import android.graphics.Bitmap;
import android.os.AsyncTask;
import android.util.Log;
import android.widget.TextView;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.InputStreamReader;
import java.net.Socket;
import java.nio.charset.StandardCharsets;

public class GetResFromServ extends AsyncTask<Bitmap, Void, String> {

    public String HOST = "";
    public int PORT = 8081;
    public static int PKT_LEN = 1024;

    @SuppressLint("StaticFieldLeak")
    private final TextView statusTxt;
    @SuppressLint("StaticFieldLeak")
    private final TextView debugTxt;

    GetResFromServ(TextView statusTxt, TextView debugTxt, String serverInfo) {
        this.statusTxt = statusTxt;
        this.debugTxt = debugTxt;
        try{
            String[] server = serverInfo.split(":");
            HOST = server[0];
            PORT = Integer.parseInt(server[1]);
        } catch (Exception e) {
            this.debugTxt.setText("ERROR: Server information wrong.");
            e.printStackTrace();
        }
    }

    @Override
    protected String doInBackground(Bitmap... bitmaps) {
        String label = "";
        try {
            Socket socket = new Socket(HOST, PORT);

            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            bitmaps[0].compress(Bitmap.CompressFormat.PNG, 100, baos);
            byte[] byteArray = baos.toByteArray();
            Log.v("perturbed bitmap ByteArray length", byteArray.length+"");

            DataOutputStream out = new DataOutputStream(socket.getOutputStream());
            int size = byteArray.length;
            StringBuilder sb = new StringBuilder(String.valueOf(size));
            while(sb.length() < 16) {
                sb.append(" ");
            }
            byte[] imgSizeBytes = sb.toString().getBytes(StandardCharsets.UTF_8);
            out.write(imgSizeBytes);
            out.flush();

            int start = 0;
            while (size != start) {
                int data_len = Math.min(size - start, PKT_LEN);
                out.write(byteArray, start, data_len);
                out.flush();
                start = Math.min(start + PKT_LEN, size);
            }

            // receive classification result
            BufferedReader br = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            char[] recvBuffer = new char[1];
            br.read(recvBuffer, 0, 1);
            label = String.valueOf(recvBuffer).trim();
            out.close();
            socket.close();

        } catch (Exception e) {
            debugTxt.setText("ERROR: See log for details.");
            e.printStackTrace();
        }
        return label;
    }

    @Override
    protected void onPostExecute(String label) {
        statusTxt.setText("Predicted label from server: " + ImageClasses.CIFAR10_CLASSES[Integer.parseInt(label)]);
    }
}
