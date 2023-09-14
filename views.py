# To Do
#   Better error checking on invalid uploads
#   Write R code to reproduce this from downloaded raw data


import json
import ast
import csv
import os
from itertools import cycle
from plotly.express import colors as plotly_colors

from io import StringIO

import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.stats import circmean, circstd

from trimesh.visual.color import hex_to_rgba
from trimesh import Trimesh
from trimesh.exchange.ply import export_ply

from django.shortcuts import HttpResponse, render
from django.middleware.csrf import get_token
from django.core.files import File

from osa.views import add_section

from src.views import clean_df_column_names

from myproject.settings import STATIC_ROOT


def orientations(request):
    _ = get_token(request)
    context = {}
    context = {**context, **add_section(request, 'tech')}
    context['title'] = 'Orientations Analysis'
    context['errors'] = []
    return render(request, 'orientations/orientations.html', context)


def orientations_r(request):
    context = {}
    return render(request, 'orientations/orientations_in_r.html')


def get_database(section):
    return


def write_file(filename, file):
    with open(filename, 'wb') as f:
        myfile = File(f)
        for chunk in file:
            myfile.write(chunk)


def title_case_columns(data):
    title_case = {name: name.title() for name in list(data.columns)}
    return data.rename(columns=title_case)


def are_valid(data):
    if all(name in list(data.columns) for name in ['X', 'Y', 'Z']):
        return {'valid': True}
    if all(name in list(data.columns) for name in ['X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2']):
        return {'valid': True}
    if all(name in list(data.columns) for name in ['Bearing', 'Plunge']):
        return {'valid': True}
    return {'valid': False, 'Message': 'Error: Missing columns called either X, Y, and Z or X1, Y1, Z1, X2, Y2, and Z2 or Plunge and Bearing.'}


def interleave_with_spacer(x1, x2):
    spacer = [None] * len(x1)
    return [val for pair in zip(x1, x2, spacer) for val in pair]


def get_color_by_fields(df):
    color_by = []
    for column in list(df.select_dtypes(include='object').columns):
        if len(df[column].unique()) < 36:
            color_by.append(column)
    return color_by


def clean_pandas_html_table(html):
    html = html.replace('border="1"', '')
    html = html.replace('dataframe', 'table')
    html = html.replace('style="text-align: right;"', '')
    return html


def convert_p_and_b_to_xyz(data):
    data['X1'] = 0
    data['Y1'] = 0
    data['Z1'] = 0
    data['X2'] = np.sin(np.radians(data['Bearing']))
    data['Y2'] = np.cos(np.radians(data['Bearing']))
    data['Z2'] = -np.tan(np.radians(data['Plunge']))
    return data


def callback(request, endpoint):

    processed_data = pd.DataFrame()
    benn_results = pd.DataFrame(columns=['Isotrophy', 'Elongation', 'Residual', 'N'])
    circstat_results = pd.DataFrame(columns=['Bearing Mean', 'Bearing SD', 'Plunge Mean', 'Plunge SD', 'N'])
    filename = None
    color_field = "All"

    if 'filename' in request.POST:
        filename = request.POST['filename']
        write_file(filename, request.FILES['file'].file)

    elif endpoint == 'demo':
        filename = os.path.join(STATIC_ROOT, 'osa/orientations/cc-a1-2-shots-clean.csv')

    elif endpoint == 'demodownload':
        filename = os.path.join(STATIC_ROOT, 'osa/orientations/cc-a1-2-shots-clean.csv')
        raw_data = pd.read_csv(filename)
        return HttpResponse(json.dumps({'data': raw_data.to_csv(quoting=csv.QUOTE_NONNUMERIC, lineterminator='\r\n', index=False)}), content_type='application/json')

    elif endpoint == 'recolor':
        body = ast.literal_eval(request.body.decode('UTF-8'))
        processed_data = pd.read_csv(StringIO(body['data']))
        color_field = body['color_by']

    elif endpoint == 'plydownload':
        body = ast.literal_eval(request.body.decode('UTF-8'))
        processed_data = pd.read_csv(StringIO(body['data']))
        color_field = body['color_by']
        ply = make_ply(processed_data, color_field)
        return HttpResponse(ply, content_type='application/json')

    if filename:
        try:
            raw_data = pd.read_csv(filename)            # Need error trapping here for bad csv file
            if 'osa/orientations/cc-a1-2-shots-clean.csv' not in filename:
                os.remove(filename)
        except [UnicodeDecodeError, FileNotFoundError] as e:
            if 'osa/orientations/cc-a1-2-shots-clean.csv' not in filename:
                os.remove(filename)
            return HttpResponse(json.dumps({'error': "Error: Invalid CSV file."}), content_type='application/json')

        raw_data = title_case_columns(raw_data)
        if are_valid(raw_data)['valid']:
            if 'X1' not in raw_data.columns and 'X' not in raw_data.columns:
                processed_data = convert_p_and_b_to_xyz(raw_data)
            elif 'X1' not in raw_data.columns:
                raw_data = only_2shots(raw_data)
                processed_data = convert_from_line_by_line(raw_data)
            else:
                processed_data = raw_data
            processed_data['All'] = 'All'           # Add a column to color by
        else:
            return HttpResponse(json.dumps({'error': are_valid(raw_data)['Message']}), content_type='application/json')

    if not processed_data.empty:
        processed_data = schdmit_statistics(processed_data)

        results = {"points": {}, "options": {}, "schmidt": {}, "benn": {}, "rose": {}}

        color_field_type = "CharField"

        if color_field_type in ['CharField', 'IntegerField']:
            processed_data[color_field].replace(np.nan, 'N/A', inplace=True)
            for color in sorted(processed_data[color_field].unique()):
                # Check to see if these are plunge bearing data only and if so skip plotting the points
                if processed_data['X1'].sum() == 0:
                    results["points"][color] = {'x': [],
                                                'y': [],
                                                'z': [],
                                                'text': []}
                else:
                    if 'Squid' in processed_data.columns:
                        text = interleave_with_spacer(list(processed_data[processed_data[color_field] == color].Squid.values), list(processed_data[processed_data[color_field] == color].Squid.values))
                    else:
                        text = interleave_with_spacer(list(processed_data[processed_data[color_field] == color].index), list(processed_data[processed_data[color_field] == color].index))
                    results["points"][color] = {'x': interleave_with_spacer(list(processed_data[processed_data[color_field] == color].X1.values.astype(float)), list(processed_data[processed_data[color_field] == color].X2.values.astype(float))),
                                                'y': interleave_with_spacer(list(processed_data[processed_data[color_field] == color].Y1.values.astype(float)), list(processed_data[processed_data[color_field] == color].Y2.values.astype(float))),
                                                'z': interleave_with_spacer(list(processed_data[processed_data[color_field] == color].Z1.values.astype(float)), list(processed_data[processed_data[color_field] == color].Z2.values.astype(float))),
                                                'text': text}

                isotrophy, elongation, residual = benn_statistics(processed_data[processed_data[color_field] == color])
                benn_results.loc[color] = [isotrophy, elongation, residual, len(processed_data[processed_data[color_field] == color])]
                circstat_results.loc[color] = [round(circmean(processed_data[processed_data[color_field] == color].bearing.values, high=360), 3),
                                                round(circstd(processed_data[processed_data[color_field] == color].bearing.values, high=90), 3),
                                                round(circmean(processed_data[processed_data[color_field] == color].plunge.values, high=360), 3),
                                                round(circstd(processed_data[processed_data[color_field] == color].plunge.values, high=90), 3),
                                                len(processed_data[processed_data[color_field] == color])]
                if 'Squid' in processed_data.columns:
                    text = list(processed_data[processed_data[color_field] == color].Squid.values)
                else:
                    text = ["Case " + str(i) for i in processed_data[processed_data[color_field] == color].index]
                results['schmidt'][color] = {'r': list(processed_data[processed_data[color_field] == color].r.values),
                                                'theta': list(processed_data[processed_data[color_field] == color].bearing.values),
                                                'text': text}
                results['benn'][color] = {'isotrophy': [isotrophy], 'elongation': [elongation], 'residual': [residual], 'text': color}
                results['rose'][color] = {'bearing': bin_bearings(processed_data[processed_data[color_field] == color].bearing),
                                            'plunge': bin_plunges(processed_data[processed_data[color_field] == color].plunge)}
            results['options']['color'] = 'discrete'

            for key in ['points', 'schmidt', 'benn', 'rose']:
                if results[key] == {}:
                    results.pop(key)
        results['options']['scaling'] = xyz_3d_scaling(processed_data)
        results["message"] = f"{processed_data.shape[0]} points were processed."
        results['processed_data'] = processed_data.drop(labels=['rise', 'run', 'delta_x', 'delta_y', 'delta_z'], axis=1).to_csv()
        results['color_by_fields'] = get_color_by_fields(processed_data)
        results['benn_table'] = clean_pandas_html_table(benn_results.to_html())
        results['benn_csv'] = benn_results.to_csv(quoting=csv.QUOTE_NONNUMERIC, lineterminator='\r\n')
        results['circstats_table'] = clean_pandas_html_table(circstat_results.to_html())
        results['circstats_csv'] = circstat_results.to_csv(quoting=csv.QUOTE_NONNUMERIC, lineterminator='\r\n')
        return HttpResponse(json.dumps(results), content_type='application/json')

    else:
        return HttpResponse(json.dumps({'error': "Bad endpoint"}), content_type='application/json')


def only_2shots(xyz_data):
    if 'Squid' in xyz_data.columns:
        xyz_data = xyz_data.groupby(['Squid']).filter(lambda x: len(x) == 2).reset_index()
        return xyz_data
    if 'Id' in xyz_data.columns:
        xyz_data = xyz_data.groupby(['Id']).filter(lambda x: len(x) == 2).reset_index()
        return xyz_data
    return xyz_data


def convert_from_line_by_line(xyz_data):
    xyz_data_suffix_0 = xyz_data[::2].reset_index()
    xyz_data_suffix_0.rename({'X': 'X1', 'Y': 'Y1', 'Z': 'Z1'}, inplace=True, axis=1)
    xyz_data_suffix_1 = xyz_data[1::2][['X', 'Y', 'Z']].reset_index()
    xyz_data_suffix_1.rename({'X': 'X2', 'Y': 'Y2', 'Z': 'Z2'}, inplace=True, axis=1)
    return pd.concat([xyz_data_suffix_0, xyz_data_suffix_1], axis=1).drop(labels=['Suffix', 'index'], axis=1)


def bin_bearings(bearings):
    hist, edges = np.histogram(bearings, bins=36, range=(0, 360), density=False)
    return list(hist.tolist())


def bin_plunges(plunges):
    hist, edges = np.histogram(plunges, bins=18, range=(0, 90), density=False)
    return list(hist.tolist())


def xyz_3d_scaling(xyz_data):
    x, y, z = xyz_data[['X1', 'Y1', 'Z1']].max() - xyz_data[['X1', 'Y1', 'Z1']].min() if xyz_data.shape[0] > 0 else (1, 1, 1)
    return (dict(x=1, y=y / x if x != 0 else 1, z=z / x if x != 0 else 1))


def make_ply(processed_data, color_field):
    # These are processed data with X1, Y1, Z1 and X2, Y2, Z2 as well as plunge and bearing angle in decimal degrees

    # get artifact lengths
    # make ply box
    # scale the box by artifact lengths
    # rotate box onto artifact
    color_code = '#FECB52'
    opacity = 255
    colors = []
    faces = []
    vertices = []
    vertex_count = 0
    results = {'ply': ''}

    for color, color_code in zip(sorted(processed_data[color_field].unique()), cycle(plotly_colors.qualitative.D3)):
        for index, row in processed_data[processed_data[color_field] == color].iterrows():
            # a_vial = point_to_boxdf([row.x, row.y, row.z], [size, size, size])
            artifact_length = np.sqrt((row.X1 - row.X2)**2 + (row.Y1 - row.Y2)**2 + (row.Z1 - row.Z2)**2)
            center_x = (row.X1 + row.X2) / 2
            center_y = (row.Y1 + row.Y2) / 2
            center_z = (row.Z1 + row.Z2) / 2
            a_vial = point_to_boxdf([0, 0, 0], [.0125, artifact_length / 2, .0125])
            # need to now rotate the vector 0, 0, 1 onto the artifact and apply that to the box made here
            r = R.from_euler('x', -row.plunge, degrees=True)
            rot_box = r.apply(a_vial)
            r = R.from_euler('z', -row.bearing, degrees=True)
            rot_box = r.apply(rot_box)
            a_vial = pd.DataFrame({'x': rot_box[:, 0] + center_x, 'y': rot_box[:, 1] + center_y, 'z': rot_box[:, 2] + center_z})
            vial_vertices, vial_faces = make_ply_cube(a_vial, vertex_count)
            vertices += vial_vertices
            faces += vial_faces
            vertex_count += 8
            color = hex_to_rgba(color_code)
            color[3] = opacity
            colors += [color] * 8

    points_3d = Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
    results['ply'] = export_ply(points_3d, encoding='ascii').decode('utf-8')

    return json.dumps(results)


# Not implimented yet
def make_ply_cube(vertices_pd, vertex_count):
    """
    Take a dataframe of eight corners.
    Return a set of faces representing a cube suitable for writing a PLY.
    """
    # Take a dataframe of eight corners and return a set of faces
    vertices = []
    for i in range(0, 8):
        vertices.append([vertices_pd.iloc[i]['x'], vertices_pd.iloc[i]['y'], vertices_pd.iloc[i]['z']])

    faces = [[vertex_count, vertex_count + 2, vertex_count + 1]]            # Top
    faces.append([vertex_count, vertex_count + 3, vertex_count + 2])

    faces.append([vertex_count, vertex_count + 1, vertex_count + 4])        # Back side
    faces.append([vertex_count + 1, vertex_count + 5, vertex_count + 4])

    faces.append([vertex_count + 3, vertex_count + 7, vertex_count + 2])    # Front face
    faces.append([vertex_count + 2, vertex_count + 7, vertex_count + 6])

    faces.append([vertex_count + 1, vertex_count + 2, vertex_count + 5])    # Right face
    faces.append([vertex_count + 5, vertex_count + 2, vertex_count + 6])

    faces.append([vertex_count, vertex_count + 4, vertex_count + 3])        # Left face
    faces.append([vertex_count + 7, vertex_count + 3, vertex_count + 4])

    faces.append([vertex_count + 4, vertex_count + 5, vertex_count + 6])    # Bottom face
    faces.append([vertex_count + 4, vertex_count + 6, vertex_count + 7])

    return [vertices, faces]


# Not implimented yet
def point_to_boxdf(xyz, size=[.0125, .0125, .0125]):
    """
    Take a single point and its size.
    Return a dataframe with the corners of a cube.
    """
    x = [xyz[0] - size[0], xyz[0] + size[0], xyz[0] + size[0], xyz[0] - size[0]]
    x = x + x
    y = [xyz[1] + size[1], xyz[1] + size[1], xyz[1] - size[1], xyz[1] - size[1]]
    y = y + y
    z = [xyz[2] + size[2], xyz[2] + size[2], xyz[2] + size[2], xyz[2] + size[2]]
    z = z + [xyz[2] - size[2], xyz[2] - size[2], xyz[2] - size[2], xyz[2] - size[2]]
    df = pd.DataFrame({'x': x, 'y': y, 'z': z})
    return df


# Not implimented yet
def build_ply(db, filters, color, points='one-point', is_authenticated=False):

    results = {'ply': ''}
    if 'downloads' in db:
        if db['downloads'] is False:
            return results

    """
    Take a dataframe of points with option size, color and opacity.
    Return the faces and vertices for a PLY of those points.
    Points are rendered as cubes of specified size.
    """
    color_code = '#FECB52'
    size = .0125
    opacity = 255
    colors = []
    faces = []
    vertices = []
    vertex_count = 0

    color_field, color_field_relational, color_field_type = build_colorfield(db, color)
    filter = build_filters(filters)
    filter = add_section_filters(filter, db)
    xyz_data = pd.DataFrame.from_records(db['context']['table'].objects.filter(**filter).values('xyz__x', 'xyz__y', 'xyz__z', 'xyz__suffix', 'squid', color_field_relational))

    if not xyz_data.empty:
        xyz_data = clean_df_column_names(xyz_data)
        xyz_data.dropna(subset=['x', 'y', 'z'], inplace=True)

        if points == 'one_point':
            columns = xyz_data.columns.values.tolist()
            xyz_data = xyz_data.groupby(columns[4:], as_index=False)[['x', 'y', 'z']].mean()

        elif points == '2-shots':
            xyz_data = only_2shots(xyz_data)

        if color_field_type == 'CharField':
            for color, color_code in zip(sorted(xyz_data[color_field].unique()), cycle(plotly_colors.qualitative.D3)):
                for index, row in xyz_data[xyz_data[color_field] == color].iterrows():
                    a_vial = point_to_boxdf([row.x, row.y, row.z], [size, size, size])
                    vial_vertices, vial_faces = make_ply_cube(a_vial, vertex_count)
                    vertices += vial_vertices
                    faces += vial_faces
                    vertex_count += 8
                    color = hex_to_rgba(color_code)
                    color[3] = opacity
                    colors += [color] * 8
        else:
            for index, row in xyz_data.iterrows():
                a_vial = point_to_boxdf([row.x, row.y, row.z], [size, size, size])
                vial_vertices, vial_faces = make_ply_cube(a_vial, vertex_count)
                vertices += vial_vertices
                faces += vial_faces
                vertex_count += 8
                color = hex_to_rgba(color_code)
                color[3] = opacity
                colors += [color] * 8

        points_3d = Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
        results['ply'] = export_ply(points_3d, encoding='ascii').decode('utf-8')

    return json.dumps(results)


def schdmit_statistics(xyz_data):
    """
    Code literally adapted from McPherron orientations analysis originally written in R
    Not optimized for numpy.
    """
    xyz_data['delta_x'] = xyz_data['X2'] - xyz_data['X1']
    xyz_data['delta_y'] = xyz_data['Y2'] - xyz_data['Y1']
    xyz_data['delta_z'] = xyz_data['Z2'] - xyz_data['Z1']

    xyz_data = xyz_data.drop(xyz_data[(xyz_data['delta_x'] == 0) & (xyz_data['delta_y'] == 0) & (xyz_data['delta_z'] == 0)].index)

    xyz_data['run'] = (xyz_data['delta_x'] ** 2 + xyz_data['delta_y'] ** 2) ** .5
    xyz_data['plunge'] = np.where(xyz_data['run'] == 0, 90, np.degrees(np.arctan(abs(xyz_data['delta_z']) / xyz_data['run'])))

    xyz_data['run'] = np.where(xyz_data['Z1'] > xyz_data['Z2'], xyz_data['X2'] - xyz_data['X1'], xyz_data['X1'] - xyz_data['X2'])
    xyz_data['rise'] = np.where(xyz_data['Z1'] > xyz_data['Z2'], xyz_data['Y2'] - xyz_data['Y1'], xyz_data['Y1'] - xyz_data['Y2'])

    xyz_data['bearing'] = np.where(xyz_data['run'] == 0, 0, np.degrees(np.arctan(xyz_data['rise'] / xyz_data['run'])))

    xyz_data['bearing'] = 90 - xyz_data['bearing']
    xyz_data['bearing'] = np.where(xyz_data['run'] <= 0, xyz_data['bearing'] + 180, xyz_data['bearing'])
    xyz_data['bearing'] = xyz_data['bearing'] % 360

    xyz_data['r'] = np.sin(np.radians((90 - xyz_data['plunge']) / 2)) / np.sin(np.radians(45))

    return xyz_data


def normalize(row):
    x, y, z = row[['delta_x', 'delta_y', 'delta_z']]
    length = np.sqrt(x**2 + y**2 + z**2)
    return pd.Series({'x': x / length, 'y': y / length, 'z': z / length})


def benn_statistics(xyz_data):

    if xyz_data.shape[0] <= 1:
        return (0, 1, 0)

    normal_vectors = xyz_data.apply(normalize, axis=1)

    l = normal_vectors['x'].values
    m = normal_vectors['y'].values
    n = normal_vectors['z'].values

    # Build a matrix prior to computing eigen values
    M11 = np.sum(l ** 2)
    M12 = np.sum(l * m)
    M13 = np.sum(l * n)
    M21 = np.sum(m * l)
    M22 = np.sum(m ** 2)
    M23 = np.sum(m * n)
    M31 = np.sum(n * l)
    M32 = np.sum(n * m)
    M33 = np.sum(n ** 2)
    M = np.array([[M11, M12, M13], [M21, M22, M23], [M31, M32, M33]])

    # Compute eigen values on matrix normalized for sample size
    n = normal_vectors.shape[0]
    e = np.linalg.eigvals(M / n)
    e = np.sort(e)[::-1]

    isotropy = e[2] / e[0]
    elongation = 1 - (e[1] / e[0])
    residual = 1 - isotropy - elongation

    return (round(isotropy, 3), round(elongation, 3), round(residual, 3))
