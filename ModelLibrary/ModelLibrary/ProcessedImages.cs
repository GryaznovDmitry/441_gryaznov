using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using Microsoft.ML;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using static Microsoft.ML.Transforms.Image.ImageResizingEstimator;
using System.Threading.Tasks;
using System.Threading.Tasks.Dataflow;

namespace ModelLibrary
{
    class ProcessedImages
    {
        ConcurrentBag<Bitmap> bag = new ConcurrentBag<Bitmap>();
        int countOfImages;

        public ProcessedImages(int count)
        {
            countOfImages = count;
        }

        public void Add(Bitmap bitmap)
        {
            bag.Add(bitmap);
        }
    }
}