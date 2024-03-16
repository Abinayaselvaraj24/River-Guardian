from exif import Image


def decimal_coords(coords, ref):
 decimal_degrees = coords[0] + coords[1] / 60 + coords[2] / 3600
 if ref == "S" or ref == "W":
     decimal_degrees = -decimal_degrees
 return decimal_degrees

def create_google_maps_url(latitude, longitude):
    base_url = "https://www.google.com/maps"
    query_params = f"?q={latitude},{longitude}"
    return base_url + query_params

def image_coordinates(image_path):
    with open(image_path, 'rb') as src:
        img = Image(src)






    if img.has_exif:
        img.gps_longitude
        coords = (decimal_coords(img.gps_latitude,
                                 img.gps_latitude_ref),
                  decimal_coords(img.gps_longitude,
                                 img.gps_longitude_ref))
        url = create_google_maps_url(coords[0], coords[1])
        lat = coords[0]
        longi = coords[1]

        return url, lat, longi
    else:
        return "","",""













