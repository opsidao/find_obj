#!/usr/bin/env ruby
# -*- mode: ruby; coding: utf-8 -*-

require 'opencv'
require 'benchmark'
include OpenCV

def compare_surf_descriptors(d1, d2, best, length)
  fail ArgumentError unless (length % 4) == 0
  total_cost = 0
  0.step(length - 1, 4) do |i|
    t0 = d1[i] - d2[i]
    t1 = d1[i + 1] - d2[i + 1]
    t2 = d1[i + 2] - d2[i + 2]
    t3 = d1[i + 3] - d2[i + 3]
    total_cost += t0 * t0 + t1 * t1 + t2 * t2 + t3 * t3
    break if total_cost > best
  end
  total_cost
end

def naive_nearest_neighbor(vec, laplacian, model_keypoints, model_descriptors)
  length = model_descriptors[0].size
  neighbor = nil
  dist1 = 1e6
  dist2 = 1e6

  model_descriptors.size.times do |i|
    kp = model_keypoints[i]
    mvec = model_descriptors[i]
    next if laplacian != kp.laplacian

    d = compare_surf_descriptors(vec, mvec, dist2, length)
    if d < dist1
      dist2 = dist1
      dist1 = d
      neighbor = i
    elsif d < dist2
      dist2 = d
    end
  end

  (dist1 < 0.6 * dist2) ? neighbor : nil
end

def find_pairs(object_keypoints, object_descriptors,
               image_keypoints, image_descriptors)
  ptpairs = []
  object_descriptors.size.times do |i|
    kp = object_keypoints[i]
    descriptor = object_descriptors[i]
    nearest_neighbor = naive_nearest_neighbor(descriptor, kp.laplacian,
                                              image_keypoints, image_descriptors)
    unless nearest_neighbor.nil?
      ptpairs << i
      ptpairs << nearest_neighbor
    end
  end
  ptpairs
end

def locate_planar_object(object_keypoints, object_descriptors,
                         image_keypoints, image_descriptors, src_corners)
  ptpairs = find_pairs(object_keypoints, object_descriptors, image_keypoints, image_descriptors)
  n = ptpairs.size / 2
  return nil if n < 4

  pt1 = []
  pt2 = []
  n.times do |i|
    pt1 << object_keypoints[ptpairs[i * 2]].pt
    pt2 << image_keypoints[ptpairs[i * 2 + 1]].pt
  end

  _pt1 = CvMat.new(1, n, CV_32F, 2)
  _pt2 = CvMat.new(1, n, CV_32F, 2)
  _pt1.set_data(pt1)
  _pt2.set_data(pt2)
  h = CvMat.find_homography(_pt1, _pt2, :ransac, 5)

  dst_corners = []
  4.times do |i|
    x = src_corners[i].x
    y = src_corners[i].y
    z = 1.0 / (h[6][0] * x + h[7][0] * y + h[8][0])
    x = (h[0][0] * x + h[1][0] * y + h[2][0]) * z
    y = (h[3][0] * x + h[4][0] * y + h[5][0]) * z
    dst_corners << CvPoint.new(x.to_i, y.to_i)
  end

  dst_corners
end

object_filename = (ARGV.size == 2) ? ARGV[0] : 'images/box.png'
scene_filename = (ARGV.size == 2) ? ARGV[1] : 'images/box_in_scene.png'

object, image = nil, nil
begin
  object = IplImage.load(object_filename, CV_LOAD_IMAGE_GRAYSCALE)
  image = IplImage.load(scene_filename, CV_LOAD_IMAGE_GRAYSCALE)
rescue
  puts "Can not load #{object_filename} and/or #{scene_filename}"
  puts "Usage: ruby #{__FILE__} [<object_filename> <scene_filename>]"
  exit
end

param = CvSURFParams.new(1500)

object_keypoints, object_descriptors = object.extract_surf(param)
image_keypoints, image_descriptors = image.extract_surf(param)

correspond = IplImage.new(image.width, object.height + image.height, CV_8U, 1)
correspond.set_roi(CvRect.new(0, 0, object.width, object.height))
object.copy(correspond)
correspond.set_roi(CvRect.new(0, object.height, image.width, image.height))
image.copy(correspond)
correspond.reset_roi

src_corners = [CvPoint.new(0, 0), CvPoint.new(object.width, 0),
               CvPoint.new(object.width, object.height), CvPoint.new(0, object.height)]
dst_corners = locate_planar_object(object_keypoints, object_descriptors,
                                   image_keypoints, image_descriptors, src_corners)

if dst_corners
  puts 'Found'
else
  puts 'Not found'
end
