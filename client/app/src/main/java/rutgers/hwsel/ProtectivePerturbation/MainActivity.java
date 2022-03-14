package rutgers.hwsel.ProtectivePerturbation;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import org.pytorch.MemoryFormat;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.util.Random;

import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {

  public static float[] CIRFAR10_NORM_MEAN_RGB = new float[] {0.4914f, 0.4822f, 0.4465f};
  public static float[] CIRFAR10_NORM_STD_RGB = new float[] {0.2471f, 0.2435f, 0.2616f};
  public static int WIDTH = 32;
  public static int HEIGHT = 32;
  public static int DATASET_SIZE = 5000;
  public static String MODEL_FOLDER = "model";
  public static String TARGET_MODEL_FOLDER = "target_model";
  public static String DATA_FOLDER = "cifar10";
  public static String TIME_RESULT_FILE = "time_result.txt";

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    final TextView statusTxt = findViewById(R.id.statusTxt);
    statusTxt.setText("Waiting choice...");
    final TextView debugTxt = findViewById(R.id.debugTxt);
    debugTxt.setText("[DEBUG] Always check Logcat for detailed log.");
    debugTxt.setEllipsize(TextUtils.TruncateAt.valueOf("START"));
    debugTxt.setSingleLine(true);

    final String[] modelList = getModelList(getApplicationContext(), MODEL_FOLDER);
    if (modelList == null) {
      debugTxt.setText("Error: cannot find models.");
      return;
    }

    final String[] imgList = getImgList(getApplicationContext());
    if (imgList == null) {
      debugTxt.setText("Error: cannot find img.");
      return;
    }

    Button demoBtn = findViewById(R.id.demoBtn);
    demoBtn.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View view) {
        int imgIndex = new Random().nextInt(DATASET_SIZE);

        String pertGenModelName = modelList[25];
        String[] submodels = pertGenModelName.substring(0, modelList[25].length()-4).split("-X-");

        Module pertGenModel = null;
        Bitmap bitmap = null;
        Module targetModel = null;
        try {
          Log.v("protective perturbation generation model name", pertGenModelName);
          Log.v("img chosen", imgList[imgIndex]);
          pertGenModel = LiteModuleLoader.load(assetModelPath(getApplicationContext(), MODEL_FOLDER, pertGenModelName));
          bitmap = BitmapFactory.decodeStream(getAssets().open(DATA_FOLDER + "/" + imgList[imgIndex]));
          targetModel = LiteModuleLoader.load(assetModelPath(getApplicationContext(), TARGET_MODEL_FOLDER, submodels[0]+".ptl"));
        } catch (IOException e) {
          Log.e("Protective Perturbation Demo", "Error reading assets.", e);
          debugTxt.setText("Error: model or image not found.");
          return;
        }

        statusTxt.setText("Waiting response from server...");
        TextView oriTxt = findViewById(R.id.oriTxt);
        oriTxt.setVisibility(View.VISIBLE);
        TextView protTxt = findViewById(R.id.protTxt);
        protTxt.setVisibility(View.VISIBLE);

        ImageView imageView = findViewById(R.id.image);
        imageView.setImageBitmap(bitmap);

        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
                CIRFAR10_NORM_MEAN_RGB, CIRFAR10_NORM_STD_RGB, MemoryFormat.CHANNELS_LAST);

        final Tensor outputTensor = pertGenModel.forward(IValue.from(inputTensor)).toTensor(); /// 1x3x32x32
        Bitmap perturbedBmp = getBitmapFromTensor(outputTensor);
        ImageView imageView2 = findViewById(R.id.image2);
        imageView2.setImageBitmap(perturbedBmp);

        TextView serverInfoTxt = findViewById(R.id.serverInfoTxt);
        String serverInfo = serverInfoTxt.getText().toString();
        GetResFromServ getResFromServ = new GetResFromServ(statusTxt, debugTxt, serverInfo);
        getResFromServ.execute(perturbedBmp);

        final Tensor pred = targetModel.forward(IValue.from(outputTensor)).toTensor();
        float[] scores = pred.getDataAsFloatArray();
        float maxScore = -Float.MAX_VALUE;
        int maxScoreIdx = -1;
        for (int i = 0; i < scores.length; i++) {
          if (scores[i] > maxScore) {
            maxScore = scores[i];
            maxScoreIdx = i;
          }
        }

        debugTxt.setText("[DEBUG] " + imgList[imgIndex] + ". Label predicted on client " + maxScoreIdx);

        Log.v("predicted label on client", maxScoreIdx+"");

      }
    });


    final Button genPicBtn = findViewById(R.id.genPicBtn);
    genPicBtn.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View view) {
        statusTxt.setText("Generating all imgs. Take hours, dont use phone.");
        genPicBtn.setText("CLOSE THE APP TO STOP");
        new Thread() {
          @Override
          public void run() {
            for (final String modelName: modelList) {
              Log.v("protective perturbation result collecting", modelName);
              runOnUiThread(new Runnable() {
                @Override
                public void run() {
                  debugTxt.setText("Generating pics, " + modelName);
                }
              });
              addPerturbation(getApplicationContext(), modelName, imgList, true);
            }

            runOnUiThread(new Runnable() {
              @Override
              public void run() {
                debugTxt.setText("Generating pics done.");
              }
            });
          }
        }.start();
      }
    });

    final Button getTimeBtn = findViewById(R.id.getTimeBtn);
    getTimeBtn.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View view) {
        statusTxt.setText("Measuring model running time.");
        debugTxt.setText("See README for accurate measurement.");
        getTimeBtn.setText("CLOSE THE APP TO STOP");
        new Thread() {
          @Override
          public void run() {
            // modify the loop if you want to measure the time one by one and get more accurate results
            for (int i = 0; i < 64; i += 8) {
              String modelName = modelList[i];
              Log.v("protective perturbation result collecting", modelName);
              writeToTxt(getApplicationContext(), modelName, TIME_RESULT_FILE);
              long startTime = System.currentTimeMillis();
              addPerturbation(getApplicationContext(), modelName, imgList, false);
              long endTime = System.currentTimeMillis();
              writeToTxt(getApplicationContext(), String.valueOf(endTime-startTime), TIME_RESULT_FILE);
              Log.v("running time for " + modelName, String.valueOf(endTime-startTime));
            }

            runOnUiThread(new Runnable() {
              @Override
              public void run() {
                debugTxt.setText("Time measurement done.");
              }
            });
          }
        }.start();
      }
    });
  }


  private static String[] getModelList(Context context, String subFolder) {
    String[] modelList = null;
    try {
      modelList = context.getAssets().list(subFolder);
    } catch (IOException e) {
      Log.e("Protective Perturbation Demo", " loading model list", e);
    }
    return modelList;
  }


  private static String[] getImgList(Context context) {
    String[] imgList = null;
    try {
      imgList = context.getAssets().list(DATA_FOLDER);
    } catch (IOException e) {
      Log.e("Protective Perturbation Demo", " loading img list", e);
    }
    return imgList;
  }


  private static Bitmap getBitmapFromTensor(Tensor tensor) {
    float[] newimage = tensor.getDataAsFloatArray();
    int[] pxl = new int[WIDTH*HEIGHT];
    int j = 0;
    for(int i = 0; i < WIDTH*HEIGHT*3; i=i+3){
      float r = newimage[i] * CIRFAR10_NORM_STD_RGB[0] + CIRFAR10_NORM_MEAN_RGB[0];
      float g = newimage[i+1] * CIRFAR10_NORM_STD_RGB[1] + CIRFAR10_NORM_MEAN_RGB[1];
      float b = newimage[i+2] * CIRFAR10_NORM_STD_RGB[2] + CIRFAR10_NORM_MEAN_RGB[2];
      pxl[j++] = Color.rgb(r, g, b);
    }
    return Bitmap.createBitmap(pxl, WIDTH, HEIGHT, Bitmap.Config.RGB_565);
  }


  private static String assetModelPath(Context context, String subFolder, String assetName) throws IOException {
    File file = new File(context.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    assetName = subFolder + "/" + assetName;
    try (InputStream is = context.getAssets().open(assetName)) {
      try (OutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();
      }
      return file.getAbsolutePath();
    }
  }


  private static void writeToTxt(Context context, String str, String fileName) {
    String save_path = context.getExternalFilesDir("").toString();
    fileName = save_path + "/" + fileName;
    str = str + "\r\n";
    try {
      File file = new File(fileName);
      if (!file.exists()) {
        file.getParentFile().mkdirs();
        file.createNewFile();
      }
      RandomAccessFile raf = new RandomAccessFile(file, "rwd");
      raf.seek(file.length());
      raf.write(str.getBytes(StandardCharsets.UTF_8));
      raf.close();
    } catch (Exception e) {
      e.printStackTrace();
    }
  }


  private static void saveBitmap(Context context, String modelName, String imgName, Bitmap bitmap) {
    String save_path = context.getExternalFilesDir(modelName).toString();
    final String filename = save_path + "/" + imgName;
    final Bitmap finalBitmap = bitmap;
    new Thread(new Runnable() {
      @Override
      public void run() {
        try (FileOutputStream out = new FileOutputStream(filename)) {
          finalBitmap.compress(Bitmap.CompressFormat.PNG, 100, out);
        } catch (IOException e) {
          e.printStackTrace();
        }
      }
    }).start();
  }


  private static void addPerturbation(Context context, String modelName, String[] imgList,
                                      boolean saveImg) {
    Bitmap bitmap = null;
    Module module = null;
    try {
      module = LiteModuleLoader.load(assetModelPath(context, MODEL_FOLDER, modelName));
    } catch (IOException e) {
      Log.e("Protective Perturbation Demo", "Error reading assets", e);
    }

    for (String img : imgList) {
      Log.v("adding perturbations to ", modelName + " " + img);
      try {
        bitmap = BitmapFactory.decodeStream(context.getAssets().open(DATA_FOLDER + "/" + img));
      } catch (IOException e) {
        Log.e("Protective Perturbation Demo", "Error reading assets" + img, e);
      }

      Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
              CIRFAR10_NORM_MEAN_RGB, CIRFAR10_NORM_STD_RGB, MemoryFormat.CHANNELS_LAST);
      Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor(); /// 1x3x32x32


      if (saveImg) {
        bitmap = getBitmapFromTensor(outputTensor);
        saveBitmap(context, modelName, img, bitmap);
      }

    }
  }

}
