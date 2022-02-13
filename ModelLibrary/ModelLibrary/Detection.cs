using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using static Microsoft.ML.Transforms.Image.ImageResizingEstimator;
using System.Threading.Tasks.Dataflow;
using System.Threading.Tasks;


namespace ModelLibrary
{
    public class Detection
    {
        // model is available here:
        // https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov4
        const string modelPath = @"C:\prac\441_gryaznov\yolov4.onnx";

        const string imageFolder = @"C:\prac\441_gryaznov\Assets\Images";

        const string imageOutputFolder = @"C:\prac\441_gryaznov\Assets\Output";

        static readonly string[] classesNames = new string[] { "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };

        public static async Task Detect()
        {
            
            MLContext mlContext = new MLContext();
            Directory.CreateDirectory(imageOutputFolder);
            // model is available here:
            // https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov4

            // Define scoring pipeline
            var pipeline = mlContext.Transforms.ResizeImages(inputColumnName: "bitmap", outputColumnName: "input_1:0", imageWidth: 416, imageHeight: 416, resizing: ResizingKind.IsoPad)
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input_1:0", scaleImage: 1f / 255f, interleavePixelColors: true))
                .Append(mlContext.Transforms.ApplyOnnxModel(
                    shapeDictionary: new Dictionary<string, int[]>()
                    {
                        { "input_1:0", new[] { 1, 416, 416, 3 } },
                        { "Identity:0", new[] { 1, 52, 52, 3, 85 } },
                        { "Identity_1:0", new[] { 1, 26, 26, 3, 85 } },
                        { "Identity_2:0", new[] { 1, 13, 13, 3, 85 } },
                    },
                    inputColumnNames: new[]
                    {
                        "input_1:0"
                    },
                    outputColumnNames: new[]
                    {
                        "Identity:0",
                        "Identity_1:0",
                        "Identity_2:0"
                    },
                    modelFile: modelPath, recursionLimit: 100));

            // Fit on empty list to obtain input data schema
            var model = pipeline.Fit(mlContext.Data.LoadFromEnumerable(new List<YoloV4BitmapData>()));

            // Create prediction engine
            var predictionEngine = mlContext.Model.CreatePredictionEngine<YoloV4BitmapData, YoloV4Prediction>(model);

            // save model
            //mlContext.Model.Save(model, predictionEngine.OutputSchema, Path.ChangeExtension(modelPath, "zip"));
            var sw = new Stopwatch();
            sw.Start();
           
            string[] imageName = Directory.GetFiles(imageFolder);


            var ab1 = new ActionBlock<int>(async x =>
            {
                using (var bitmap = new Bitmap(Image.FromFile(Path.Combine(imageFolder, imageName[x]))))
                {
                    // predict
                    Console.WriteLine($"Начата обработка изображения {x + 1}");
                    var predict = predictionEngine.Predict(new YoloV4BitmapData() { Image = bitmap });
                    var results = predict.GetResults(classesNames, 0.3f, 0.7f);

                    using (var g = Graphics.FromImage(bitmap))
                    {
                        var ab2 = new ActionBlock<YoloV4Result>(async y =>
                        {
                                // draw predictions
                                var x1 = y.BBox[0];
                            var y1 = y.BBox[1];
                            var x2 = y.BBox[2];
                            var y2 = y.BBox[3];
                            g.DrawRectangle(Pens.Red, x1, y1, x2 - x1, y2 - y1);
                            using (var brushes = new SolidBrush(Color.FromArgb(50, Color.Red)))
                            {
                                g.FillRectangle(brushes, x1, y1, x2 - x1, y2 - y1);
                            }

                            g.DrawString(y.Label + " " + y.Confidence.ToString("0.00"),
                                            new Font("Arial", 12), Brushes.Blue, new PointF(x1, y1));
                            Console.WriteLine($"[{x1}] [{x2}] [{y1}] [{y2}] : Объект - {y.Label} на фото номер {x + 1}");
                            var r1 = new Random();
                            await Task.Delay(r1.Next(1000));

                        });
                        Parallel.ForEach(results, res => ab2.Post(res));
                        ab2.Complete();
                        await ab2.Completion;
                        var r = new Random();
                        Console.WriteLine($"Закончена обработка изображения {x + 1}");
                        await Task.Delay(r.Next(500));
                    
                        string FileName = Path.GetFileName(imageName[x]);
                        string SavePath = Path.Combine(imageOutputFolder, Path.ChangeExtension(FileName, "_processed" + Path.GetExtension(FileName)));
                        
                        bitmap.Save(SavePath);
                    }
                }
            });
            Parallel.For(0, imageName.Length, i => ab1.Post(i));
            ab1.Complete();

            await ab1.Completion;
            sw.Stop();
            Console.WriteLine($"Done in {sw.ElapsedMilliseconds}ms.");
        }
    }
}
