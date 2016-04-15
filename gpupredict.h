struct line_break
{
 __host__ __device__
 bool operator()(char x)
 {
   return x == '\n';
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

/*struct create_bb
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
        thrust::uniform_real_distribution<double> dist(0,1);
        curandState s;
        curand_init(i, 0, 0, &s);
        double latitude = curand_uniform(&s) + 38.4304;
        double latitude_diff = curand_uniform(&s) * .1 - .05;
        double longitude = curand_uniform(&s) * 3.08 + -79.4684;
        double longitude_diff = curand_uniform(&s) * .1 - .05;
        int month = 
        
        
        double latitude;
        double latitude_diff;
        double longitude;
        double longitude_diff;
        
    }
};
*/

/*struct gpu_atoi
{
    char *source;
    int *dest;
    int *length;
    
    gpu_atoi(char *_source, int *_dest, int *_length):
        source(_source), dest(_dest), length(_length) {}
        
    template <typename IndexType>
    
    __host__ __device__
    void operator()(IndexType & i) {
        int value = 0;  //integer value being determined
        int j = *length * i;    //index of character we are evaluating
        int max = *length * (i + 1);
        bool neg;             
        
        if (source[j] == '-') {
            neg = true;
            j++;
        } else {
            neg = false;
            if (source[j] == '+')
                j++;
        }        
        
        while(j < max && source[j] >= '0' && source[j] <= '9') {
            value *= 10;
            value += (source[j++] - '0'); 
        }
        if (neg){
            value = -value;
        }
        dest[i] = value;
    }
};*/
