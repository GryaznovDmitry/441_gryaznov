using System.Collections.Generic;
using System.Collections;


namespace Lab2
{
    public class Results : IEnumerable<string>
    {
        public string Info { get; set; }
        public List<string> images { get; set; }

        public Results(string newType, string newImage)
        {
            Info = newType;
            images = new List<string>();
            images.Add(newImage);
        }

        public override string ToString()
        {
            return Info;
        }

        public IEnumerator<string> GetEnumerator()
        {
            return ((IEnumerable<string>)images).GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return ((IEnumerable)images).GetEnumerator();
        }
    }
}
