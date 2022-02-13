using ModelLibrary;

namespace lab1
{
    public class program
    {
        public static async Task Main()
        {
            await Detection.Detect();
            Console.WriteLine("Ready");          
        }
    }
}