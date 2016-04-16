#include <thrust/random.h>
#define NUMBER_OF_BOUNDING_BOXES 300
#define EVENT_WIDTH 32

struct line_break
{
 __host__ __device__
 bool operator()(char x)
 {
   return x == '\n';
 }
};

struct null_space
{
 __host__ __device__
 bool operator()(char x)
 {
   return x != 0;
 }
};

struct column_split
{
    char *text;         //Source csv file
    int *linebreaks;    //Location of linebreaks
    char **columns;     //Destination vectors
    int *column_width;  //Size of each entry for each vector
    int *num_columns;   //Number of columns to split
    int *column_locations;  //Specifies columns to be copied
    
    column_split(char *_text, int *_linebreaks, char **_columns, int *_column_width, int *_num_columns, int *_column_locations):
    text(_text), linebreaks(_linebreaks), columns(_columns), column_width(_column_width), num_columns(_num_columns), column_locations(_column_locations) {}
    
    template <typename IndexType>
    __host__ __device__
    void operator()(const IndexType & i) {
        int column_src = 0;
        int column_dst = 0;
        int pos_src = linebreaks[i];
        int pos_dst = 0;
        
        while(column_dst < *num_columns)
        {
            if (column_locations[column_dst] == column_src) {
                pos_dst = 0;
                pos_src++;
                while(text[pos_src] != ',' && pos_dst < column_width[column_dst]) {
                    columns[column_dst][column_width[column_dst] * i + pos_dst] = text[pos_src];
                    pos_src++;
                    pos_dst++;
                }
                column_dst++;
            }
            else {
                pos_src++;
                while(text[pos_src] != ',')
                    pos_src++;
            }
            column_src++;
        }
    }
};

struct gpu_atof
{
    char *source;
    double *dest;
    int *length;
    
    gpu_atof(char *_source, double *_dest, int *_length):
    source(_source), dest(_dest), length(_length) {}
    
    template <typename IndexType>
    __host__ __device__
    void operator()(const IndexType & i) {
        double value = 0;
        double power = 10.0;
        int j = *length * i;
        int max = *length * (i + 1);
        bool neg = false;
        
        if (source[j] == '-') {
            neg = true;
            j++;
        } else {
            neg = false;
            if (source[j] == '+')
                j++;
        }
        
        while(j < max && source[j] >= '0' && source[j] <= '9') {
            value = value * 10.0 + (source[j] - '0');
            j++;
        }
        
        if (source[j] == '.') {
            j++;
            while (j < max && source[j] >= '0' && source[j] <= '9') {
                value += (source[j] - '0') / power;
                power *= 10.0;
                j++;
            }
        }
        if (neg)
            value = -value;
        dest[i] = value;
    }
};

struct get_month
{
    char *source;
    int *dest;
    
    get_month(char *_source, int *_dest):
    source(_source), dest(_dest) {}
    
    template <typename IndexType>
    __host__ __device__
    void operator()(const IndexType & i) {
        int j = 22 * i + 5;
        int month = 0;
        while(source[j] >= '0' && source[j] <= '9') {
            month = month * 10 + (source[j] - '0');
            j++;
        }
        dest[i] = month;
    }
};

struct create_bb
{
    double *min_lat;
    double *max_lat;
    double *min_long;
    double *max_long;
    int *month;
    
    create_bb(double *_min_lat, double *_max_lat, double *_min_long, double *_max_long, int *_month):
    min_lat(_min_lat), max_lat(_max_lat), min_long(_min_long), max_long(_max_long), month(_month) {}
    
    template <typename IndexType>
    __host__ __device__
    void operator()(const IndexType & i) {
        thrust::default_random_engine rng;
        thrust::uniform_real_distribution<double> double_dist;
        thrust::uniform_int_distribution<int> int_dist(1, 12);
        rng.discard(5 * i);
        min_lat[i] = double_dist(rng) + 38.4304;
        max_lat[i] = min_lat[i] + double_dist(rng) * .05;
        min_long[i] = double_dist(rng) * 3.0805 + -79.4684;
        max_long[i] = min_long[i] + double_dist(rng) * .05;
        month[i] = int_dist(rng);
    }
};

struct bb_events
{
    int events [NUMBER_OF_BOUNDING_BOXES];

    void initialize()
    {
      for (int i = 0; i < NUMBER_OF_BOUNDING_BOXES; i++)
      {
          events[i] = 0;
      }
    }
};

struct get_events
{
    double *latitude;
    double *longitude;
    int *month;
    char *event;
    double *min_lat;
    double *max_lat;
    double *min_long;
    double *max_long;
    int *bb_month;
    
    get_events(double *_latitude, double *_longitude, int *_month, char *_event, double *_min_lat, double *_max_lat, double *_min_long, double *_max_long, int *_bb_month):
    latitude(_latitude), longitude(_longitude), month(_month), event(_event), min_lat(_min_lat), max_lat(_max_lat), min_long(_min_long), max_long(_max_long), bb_month(_bb_month) {}
    
    template <typename IndexType>
    __host__ __device__
    bb_events operator()(const IndexType & i) const
    {
        bb_events result;
        for (int j = 0; j < NUMBER_OF_BOUNDING_BOXES; j++) {
            result.events[j] = 0;
            if (latitude[i] > min_lat[j] && latitude[i] < max_lat[j] && longitude[i] > min_long[j] && longitude[i] < max_long[j] && month[i] == bb_month[j] && event[i * EVENT_WIDTH] == 'a') 
                result.events[j]++;
        }
        return result;
    }
};


struct combine_events 
    : public thrust::binary_function<const bb_events&, 
                                     const bb_events&,
                                           bb_events>
{
    __host__ __device__
    bb_events operator()(const bb_events &x, const bb_events &y) const
    {
        bb_events result;
        for(int i = 0; i < NUMBER_OF_BOUNDING_BOXES; i++) {
            result.events[i] = 0;
            result.events[i] = x.events[i] + y.events[i];
        }
        return result;
    }
};

struct gpu_ftoa
{
    double *source;
    char *dest;
    int *length;
    
    gpu_ftoa(double *_source, char *_dest, int *_length):
    source(_source), dest(_dest), length(_length) {}
    
    template <typename IndexType>
    __host__ __device__
    void operator()(const IndexType & i) {
        int value_int = (int) source[i];
        double value_float = source[i] - (double) value_int;
        char *text= new char[*length];

        bool neg = false;        
        if (value_int < 0) {
            neg = true;
            value_int = -value_int;
            value_float = -value_float;
        }
        
        int j = 0;
        do {
            text[j++] = value_int % 10 + '0';
        } while ((value_int /= 10) > 0);
            
        if (neg)
            text[j++] = '-';
        
        int k = *length * i;
        int bound = *length * (i + 1);
        while(j>0 && k < bound)
            dest[k++] = text[--j];
        if(k < bound)
            dest[k++] = '.';
        for(int a = 0; a <= bound - k; a++) {
            value_float *= 10;
        }
        value_int = (int) value_float;
        
        j = 0;
        do {
            text[j++] = value_int % 10 + '0';
        } while ((value_int /= 10) > 0);


        while(j>0 && k < bound)
            dest[k++] = text[--j];
        
    }
};

struct gpu_itoa
{
    int *source;
    char *dest;
    int *length;
    
    gpu_itoa(int *_source, char *_dest, int *_length):
        source(_source), dest(_dest), length(_length) {}
        
    template <typename IndexType>
    
    __host__ __device__
    void operator()(IndexType & i) {
        int value = source[i];
        char text[33];
        bool isNegative = false;
        if (value < 0) {
            isNegative = true;
            value = -value;
        }

        int j = 0;
        do {
            text[j++] = value % 10 + '0';
        } while ((value /= 10) > 0);
            
        if (isNegative)
            text[j++] = '-';
        
        int k = i**length;
        int bound = (i+1)**length;
        while(j>0 && k <= bound)
            dest[k++] = text[--j];
    }
};

struct gpu_output
{
    char **source;
    char *dest;
    int *column_width;
    int *length;
    int *num_columns;
    
    gpu_output(char **_source, char *_dest, int *_column_width, int *_length, int *_num_columns):
        source(_source), dest(_dest), column_width(_column_width), length(_length), num_columns(_num_columns) {}
        
    template <typename IndexType>
    
    __host__ __device__
    void operator()(IndexType & i) {
        int ind = *length * i;
        int j, k, max_k;
        for(j = 0; j < *num_columns; j++) {
            k = column_width[j] * i;
            max_k = column_width[j] * (i + 1);
            while(k < max_k && source[j][k] != NULL)
                dest[ind++] = source[j][k++];
            if ((j + 1) < *num_columns)
                dest[ind++] = ',';
        }
        dest[*length * (i + 1) - 1] = '\n';
    }
};